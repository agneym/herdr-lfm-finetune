#!/usr/bin/env python3
"""Generate the README comparison graph: base vs fine-tune vs frontier models.

Base + v7 adapter are scored on the pinned 120-row v8 holdout
(runs/results/eval_v8_holdout.json). The frontier models (deepseek / GLM flash)
were scored on the prior 98-row v5 holdout via eval_pi.mjs; the v8 frontier run
is still pending, so their bars are shown as history and labeled as v5 below.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

metrics = ["exact-call", "tool-selection", "off-topic (no call)"]
# v8 (120-row) numbers for base + v7; frontier models on the v5 (98-row) holdout.
models = {
    "base (untuned)":        (6.8, 25.2, 47.1),   # v8, 120 rows
    "v7 (LoRA fine-tune)":   (96.1, 97.1, 100.0), # v8, 120 rows
    "deepseek flash (v5)":   (56.1, 65.9, 50.0),  # prior 98-row holdout
    "glm 5.3 flash (v5)":    (73.2, 87.8, 50.0),  # prior 98-row holdout
}
colors = {
    "base (untuned)":       "#9aa0a6",  # gray
    "v7 (LoRA fine-tune)":  "#1a7f37",  # green — the star
    "deepseek flash (v5)":  "#4c78a8",  # blue
    "glm 5.3 flash (v5)":   "#e07b39",  # orange
}

x = np.arange(len(metrics))
n = len(models)
width = 0.19

fig, ax = plt.subplots(figsize=(9.0, 5.4))
for i, (label, vals) in enumerate(models.items()):
    offset = (i - (n - 1) / 2) * width
    bars = ax.bar(x + offset, vals, width, label=label, color=colors[label],
                  edgecolor="white", linewidth=0.6, zorder=3)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v + 1.2),
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color="#222")

ax.set_ylabel("accuracy (%)", fontsize=11)
ax.set_ylim(0, 112)
ax.set_yticks(range(0, 101, 20))
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_title("Herdr holdout accuracy (base + fine-tune on the 120-row v8 pin)",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(loc="lower right", frameon=True, fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout(rect=(0, 0.06, 1, 1))
# Footnote disambiguating the two holdout sets, so the chart isn't misread.
fig.text(0.5, 0.01,
         "base + v7: 120-row v8 holdout (runs/results/eval_v8_holdout.json).  "
         "deepseek/GLM: prior 98-row v5 holdout (v8 frontier run pending).",
         ha="center", va="bottom", fontsize=8.5, color="#555")

fig.savefig("docs/eval_comparison.svg", format="svg")
fig.savefig("docs/eval_comparison.png", format="png", dpi=180)
print("wrote docs/eval_comparison.svg and docs/eval_comparison.png")
