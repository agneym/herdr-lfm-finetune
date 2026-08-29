#!/usr/bin/env python3
"""Generate the README comparison graph: base vs fine-tune vs deepseek flash."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

metrics = ["exact-call", "tool-selection", "off-topic (no call)"]
models = {
    "base (untuned)":        (9.8, 26.8, 68.8),
    "v7 (LoRA fine-tune)":   (96.3, 96.3, 100.0),
    "deepseek flash":        (56.1, 65.9, 50.0),
}
colors = {
    "base (untuned)":      "#9aa0a6",  # gray
    "v7 (LoRA fine-tune)": "#1a7f37",  # green — the star
    "deepseek flash":      "#4c78a8",  # blue
}

x = np.arange(len(metrics))
n = len(models)
width = 0.26

fig, ax = plt.subplots(figsize=(8.4, 5.0))
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
ax.set_title("Herdr holdout accuracy (98 pinned rows)", fontsize=13, fontweight="bold", pad=12)
ax.legend(loc="lower right", frameon=True, fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()

fig.savefig("docs/eval_comparison.svg", format="svg")
fig.savefig("docs/eval_comparison.png", format="png", dpi=180)
print("wrote docs/eval_comparison.svg and docs/eval_comparison.png")
