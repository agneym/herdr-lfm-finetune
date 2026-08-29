# Run caveats

Operational and reproducibility caveats that would clutter the README. Read this
before comparing any two runs.

## v7 is a regime change — do not compare it to v6

Through v6 every training/eval row shared ONE fixed system prompt
(workspace=`w1`, pane=`w1:p1`, cwd=`/home/repo`, agent kind=hermes), which let
the model "solve" grounding by memorizing constants. v7 deterministically
rotates the system prompt over 8 contexts (workspaces w1–w5, cwds, caller
panes, agent kinds), so it tests context-free grounding. It came out **+2.4 pts**
over v6 anyway. The pinned holdout is keyed by query string only, so it still
resolves; per-run numbers and failure modes are in `runs/eval_v7_summary.md`.

## Pre-fix eval contamination (v1–v4 invalid)

All pre-fix numbers (v1–v4, and any run before commit `798b7d8`) are invalid.
Eval's `ask()` used to render `row["messages"]` (including the gold assistant
answer) with `add_generation_prompt=True`, scoring a *continuation* rather than
a from-scratch answer. Fixed to `row["messages"][:-1]` in `eval_lfm2.py` to
match `train_lfm2.py`. Re-run any adapter with the fixed `eval_lfm2.py` before
comparing.

## LFM2 target-map gotcha (torchao)

The LoRA targets are `q/k/v` + `w1/w3/w2`. LFM2 naming gotcha: the MLP
projections are `w1`/`w3`/`w2`, NOT gate/up/down_proj (those match nothing),
and attention output is `out_proj` which is SHARED with Lfm2ShortConv — there is
**no `o_proj` at all**, so the old "o_proj" target silently trained q/k/v only.
Do NOT target `conv_in_proj`/`conv_out_proj` — PEFT routes them through torchao
and crashes.

## Colab checkpoint lifecycle

- The Colab scripts write the adapter flat at `/content/lfm2_herdr_lora`; after
  reconstructing the dumped tarball locally, unpack it into `adapters/`.
- NEVER run training inside one blocking `colab exec` — an exec timeout or a dead
  keep-alive daemon lets Colab idle-prune the VM and you lose the checkpoint.
  Use `run_detached_dump.py`: nohup-detach the trainer, poll every 120 s, tick
  keep-alive every 60 s, then tar+base64-dump the checkpoint to stdout on
  completion (the VM can be reaped seconds after training finishes). Reconstruct
  locally with the snippet in `NOTES.md` / the script's docstring.
- Recommended: mirror the checkpoint to Google Drive so it survives a VM reap —
  `colab drivemount -s NAME` (interactive OAuth once), then queue
  `scripts/copy_to_drive.py` as a second `colab exec` (it waits for the
  checkpoint tarball and copies it to `/content/drive/MyDrive/herdr/`, sha256
  verified).

## Pinned holdout

`split.py` owns the split. The eval holdout is carved out first via
`pin_holdout.py` (keyed by query string, so appending training rows never shifts
it), then train/val are split from the remainder — so the three sets are
provably disjoint. Reuse it for both eval and train
(`--holdout runs/eval_v5_holdout.json`, or `--env HOLDOUT=...` on Colab). If a
holdout query changes, the pin breaks.
