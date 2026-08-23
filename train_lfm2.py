#!/usr/bin/env python3
"""train_lfm2.py — LoRA SFT of LiquidAI/LFM2-350M on the Herdr dataset.

Trains only on assistant-token positions (prompt masked out). Designed for a
single T4/L4: bf16, gradient accumulation, small batch. Saves the adapter to
--out (default lfm2_herdr_lora).

Usage:
  python train_lfm2.py --data data_lfm2.jsonl --epochs 8 --out lfm2_herdr_lora
"""
import argparse
import json
import random

import torch
from torch.utils.data import Dataset

from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          get_cosine_schedule_with_warmup)
from peft import LoraConfig, get_peft_model

MODEL_ID = "LiquidAI/LFM2-350M"


class HerdrDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=4096):
        self.examples = []
        rows = [json.loads(l) for l in open(path) if l.strip()]
        # deterministic split: last 10% = validation (same spirit as needle's split)
        rng = random.Random(0)
        order = list(range(len(rows)))
        rng.shuffle(order)
        n_val = max(1, int(len(rows) * 0.1))
        self.val_idx = set(order[-n_val:])
        for i, row in enumerate(rows):
            full_ids = tokenizer.apply_chat_template(
                row["messages"], tools=row.get("tools") or [], tokenize=True)
            if hasattr(full_ids, "keys") and "input_ids" in full_ids:
                full_ids = full_ids["input_ids"]
            if full_ids and isinstance(full_ids[0], list):
                full_ids = full_ids[0]
            full_ids = list(full_ids)
            # prompt-only ids, to locate where the assistant target begins
            prompt_ids = tokenizer.apply_chat_template(
                row["messages"][:-1], tools=row.get("tools") or [],
                tokenize=True, add_generation_prompt=True)
            if hasattr(prompt_ids, "keys") and "input_ids" in prompt_ids:
                prompt_ids = prompt_ids["input_ids"]
            if prompt_ids and isinstance(prompt_ids[0], list):
                prompt_ids = prompt_ids[0]
            prompt_ids = list(prompt_ids)

            labels = [-100] * len(full_ids)
            start = len(prompt_ids)
            for j in range(start, len(full_ids)):
                labels[j] = full_ids[j]
            full_ids = full_ids[:max_len]
            labels = labels[:max_len]
            self.examples.append((full_ids, labels))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids, labels = self.examples[idx]
        return {"input_ids": ids, "labels": labels}

    def is_val(self, idx):
        return idx in self.val_idx


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        pad = maxlen - n
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * n + [0] * pad)
    return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn))


def run_epoch(model, ds, indices, tok, args, optimizer=None, scheduler=None):
    torch.manual_seed(0)
    order = sorted(indices, key=lambda i: len(ds.examples[i][0]))  # length-sorted batches
    batches = [order[i:i + args.batch_size]
               for i in range(0, len(order), args.batch_size)]
    total, count = 0.0, 0
    for bi, batch in enumerate(batches):
        input_ids, labels, attn = collate([ds[i] for i in batch], tok.pad_token_id or 0)
        input_ids, labels, attn = (input_ids.cuda(), labels.cuda(), attn.cuda())
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        loss = out.loss
        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        total += loss.item() * len(batch)
        count += len(batch)
        if bi % 10 == 0:
            print(f"  batch {bi}/{len(batches)}  loss {loss.item():.4f}", flush=True)
    return total / max(count, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_lfm2.jsonl")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--out", default="lfm2_herdr_lora")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F811
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lconf = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM")
    model = get_peft_model(model, lconf)
    model.print_trainable_parameters()

    ds = HerdrDataset(args.data, tok)
    train_idx = [i for i in range(len(ds)) if not ds.is_val(i)]
    val_idx = [i for i in range(len(ds)) if ds.is_val(i)]
    print(f"train {len(train_idx)}  val {len(val_idx)}")

    steps_per_epoch = -(-len(train_idx) // (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, total_steps // 20), total_steps)

    best_val = float("inf")
    micro = {"n": 0}  # grad-accum counter wrapper via closure below

    def accum_epoch(indices):
        nonlocal best_val
        order = sorted(indices, key=lambda i: len(ds.examples[i][0]))
        batches = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]
        total, count = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for bi, batch in enumerate(batches):
            input_ids, labels, attn = collate([ds[i] for i in batch], tok.pad_token_id or 0)
            out = model(input_ids=input_ids.cuda(), attention_mask=attn.cuda(),
                        labels=labels.cuda())
            (out.loss / args.grad_accum).backward()
            if (bi + 1) % args.grad_accum == 0 or bi == len(batches) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total += out.loss.item() * len(batch)
            count += len(batch)
            if bi % 20 == 0:
                print(f"  batch {bi}/{len(batches)}  loss {out.loss.item():.4f}", flush=True)
        return total / max(count, 1)

    for epoch in range(args.epochs):
        model.train()
        tr = accum_epoch(train_idx)
        model.eval()
        with torch.no_grad():
            va = run_epoch(model, ds, val_idx, tok, args)
        print(f"epoch {epoch+1}/{args.epochs}  train {tr:.4f}  val {va:.4f}", flush=True)
        if va < best_val:
            best_val = va
            model.save_pretrained(args.out)
            tok.save_pretrained(args.out)
            print(f"  saved -> {args.out}", flush=True)

    print("TRAINING OK")


if __name__ == "__main__":
    main()
