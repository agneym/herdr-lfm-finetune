# Eval snapshot — v7 (REGIME CHANGE: system-prompt rotation)

Date: 2026-08-28 (Colab L4, session lfm2v10)
Checkpoint: adapters/lfm2_herdr_lora (v6 archived as adapters/lfm2_herdr_lora_v6)
  - source: runs/ckpt-lfm2v10.tar.gz (reconstructed from runs/lfm2v10_dump.log)
  - sha256: 0c2e0622abb215eb8dccbcecc2980ee1dab4039b0c081fe637e216b9a79734d3
Dataset: dataset.jsonl v7 — 804 rows, 98 off-topic (12.2%)
  - Same row set as v6; only messages[0] (system prompt) changed.

## The regime change

Through v6, every training and eval row shared ONE fixed system prompt:

    HERDR_ENV=1 / workspace=w1 / tab=w1:t1 / pane=w1:p1 /
    cwd=/home/repo / agent kind=hermes

Under that regime the model could "solve" grounding by memorizing w1:p1 and
/home/repo as constants. The 93.9% v6 exact-call number therefore did NOT
predict real-usage behavior, where the caller's workspace, pane, cwd and agent
kind are arbitrary per session.

v7 rotates the system prompt deterministically over 8 contexts (row index
modulo 8): workspaces w1–w5, cwds (/home/repo, /home/repo/proj, /srv/api,
/opt/billing, /home/agney/code/herdr, /tmp/scratch), caller panes
(w1:p1, w1:p4, w2:p1, w2:p2, w3:p1, w3:p3, w4:p2, w5:p1) and agent kinds
(hermes, claude, codex, pi, opencode, gemini, cursor, devin). Contexts are
near-uniform: 101/101/101/101/100/100/100/100 rows.

Label policy (what makes the pin still valid):
- The pinned holdout (runs/eval_v5_holdout.json) is keyed by QUERY STRING
  only. All 98 holdout queries are unchanged, so the pin resolves against the
  v7 dataset — but each holdout row now carries whatever context its
  generation-order slot assigned it. Holdout context spread:
  w1:p1/hermes 12, w1:p4/claude 12, w2:p1/pi 14, w2:p2/codex 10,
  w3:p1/opencode 13, w3:p3/gemini 10, w4:p2/cursor 11, w5:p1/devin 16.
- Labels never depend on the rotated context: explicit ids come from the
  query; "current pane" ops use current=true; ids absent from the query are
  context-derived only in the three chained rows (split→run/wait, "my other
  pane"), where the new/other pane id is derived from the context's caller
  pane (same workspace, next p#). A leak audit confirmed no other answer
  references a pane/cwd/branch not named in its query.
- One v<=6 label bug fixed en passant: "run `npm run build` in my other pane"
  was hardcoded to w1:p2 under the old fixed context ("other pane" is only
  resolvable against the caller); it is now context-derived like the chains.
  That query is NOT in the pinned holdout.

## Comparability

Same queries and labels as v6, but the eval regime now tests context-free
grounding, so numbers are not directly comparable. Pre-registered expectation
was that exact-call would DROP (delta v6−v7 = memorized context vs skill);
the result went the other way (see Results) — the model generalizes the
system-prompt → query-echo mapping rather than memorizing one environment.
Train/eval leakage from rotation is also not a concern: each row appears with
exactly one context, and the holdout contexts were assigned by generation
order, not by model performance.

## Live validation (herdr 0.8.2)

`validate_dataset.py` replayed v7 against the live server: **PASS 404 /
SKIP 250 / FAIL 23 — byte-for-byte identical status profile to the v6
report**. All 23 failures are pre-existing (22× integration_install for
cursor/copilot/devin/droid/kilo — kinds the CLI doesn't recognize yet — and
1× worktree_create.base='develop' plus the v6 off-topic rows already known).
The rotated system prompts introduced **zero** new validation failures.

## Results

Trained on Colab L4 (session lfm2v10), hyperparams unchanged from v6 (epochs
12, batch 1, grad-accum 8, lr 1e-4, LoRA r=16 alpha=32 on q/k/v + w1/w3/w2).
Val curve (poll snapshots missed epochs 3, 6, 7, 9, 10): 0.3990, 0.1795, …,
0.0944, 0.0800, …, 0.0567, …, 0.0483, 0.0482 — best val 0.0482.  NOTE: val
loss is NOT comparable across regimes (v6 hit 0.0249 on a fixed context;
rotation makes the grounding task itself harder).

Evaluated on the SAME pinned 98-row holdout (runs/eval_v5_holdout.json),
whose rows now span all 8 rotated contexts:

| model | exact-call | exact-norm | tool-selection | off-topic |
|---|---:|---:|---:|---:|
| v6 (fixed context)          | 93.9% | 93.9% | 96.3% | 100.0% |
| **v7 (rotated context)**    | **96.3% (79/82)** | **96.3%** | **96.3%** | **100.0% (16/16)** |

The headline: rotation did NOT cost accuracy — it gained +2.4 pts exact-call
over v6. The memorization worry was wrong in the best way: the model learned
to read the caller's workspace/pane/cwd from the system prompt and to ground
explicit ids in the query. Per-tool grounding is 100% on 16 of 18 tools; the
only gaps are pane_split 4/5 and pane_current 1/3. The context-derived
"other pane" row resolves correctly.

### Remaining failure modes (3)

1. **"give me a new pane on the right" (row 104).** Emits a hallucinated tool
   (`pane_create(Direction=...)`, wrong casing) — v6 also regressed here
   (predicted pane_current). "give me a new pane" variants are thin in
   training; add more create/split paraphrases for v8.
2–3. **"where am i?" / "please, where am i?" (rows 388/389, persists since
   v5).** Under-calls (emits no tool). These surface forms are held out and
   absent from training — pane_current has ~20 training forms but not these.

## Files
- make_dataset.py            — CONTEXTS + system_prompt() rotation, _sibling_pane
- dataset.jsonl              — v7 (804 rows, rotated system prompts)
- runs/eval_v5_holdout.json  — the pinned 98-row holdout (keyed by query; still valid)
- runs/lfm2v10_dump.log      — training log + base64 checkpoint dump
- runs/ckpt-lfm2v10.tar.gz   — raw checkpoint tarball (gitignored)
- runs/eval_v7_new.log       — v7 adapter on the pinned holdout
