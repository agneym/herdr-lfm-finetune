# Eval snapshot — v7 (REGIME CHANGE: system-prompt rotation)

Date: 2026-08-27 (dataset regenerated; Colab retrain + eval PENDING)
Dataset: dataset.jsonl v7 — 804 rows, 98 off-topic (12.2%)
  - SAME row set as v6 (no rows added or removed); only `messages[0]` changed.

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

⚠ Numbers on this holdout are NOT directly comparable to v6 (93.9%): same
queries and labels, but the eval regime now tests context-free grounding.
Expect exact-call to DROP on v7 even for a well-trained adapter; the delta
v6-minus-v7 is a direct estimate of how much of v6's score was memorized
context rather than skill. There is also no train/eval leakage concern from
rotation: training rows see the same context distribution, but each row
appears with exactly one context, and the model must generalize the
system-prompt → query-echo mapping.

## Live validation (herdr 0.8.2)

`validate_dataset.py` replayed v7 against the live server: **PASS 404 /
SKIP 250 / FAIL 23 — byte-for-byte identical status profile to the v6
report**. All 23 failures are pre-existing (19× integration_install for
cursor/copilot/devin/droid/kilo — kinds the CLI doesn't recognize yet — and
1× worktree_create.base='develop' plus the v6 off-topic rows already known).
The rotated system prompts introduced **zero** new validation failures.

## Results

Pending: retrain on Colab with the v7 dataset (recipe unchanged — see
NOTES / Makefile `make train`), then:

    .venv/bin/python eval_lfm2.py --adapter lfm2_herdr_lora \
        --holdout runs/eval_v5_holdout.json | tee runs/eval_v7_new.log

| model | exact-call | exact-norm | tool-selection | off-topic |
|---|---:|---:|---:|---:|
| v6 (fixed context)          | 93.9% | 93.9% | 96.3% | 100.0% |
| v7 (rotated context)        | TBD   | TBD   | TBD   | TBD    |

## Files
- make_dataset.py            — CONTEXTS + system_prompt() rotation, _sibling_pane
- dataset.jsonl              — v7 (804 rows, rotated system prompts)
- runs/eval_v5_holdout.json  — the pinned 98-row holdout (keyed by query; still valid)
