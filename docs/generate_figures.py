#!/usr/bin/env python3
"""
Generate research figures for the PCB Defect Detection project
- Performance comparison bar chart
- Ablation study results
- Parameter efficiency visualization
- Active learning round-wise accuracy (synthetic from documented results)
Outputs are saved in docs/figures/
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "figure.dpi": 140,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FIG_DIR = Path("docs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name: str):
    out = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    # Also export vector PDF alongside PNG for publication
    try:
        pdf_out = out.with_suffix('.pdf')
        plt.savefig(pdf_out, bbox_inches="tight")
        print(f"Saved {pdf_out}")
    except Exception as e:
        pass
    print(f"Saved {out}")
    plt.close()


def performance_comparison():
    # From README.md (Enhanced Active Learning Results):
    # Baseline CNN: Test Acc 19.33%
    # Enhanced ResNet+LoRA: Test Acc 20.33% (+1.00%)
    import pandas as pd

    data = pd.DataFrame(
        {
            "Model": ["Baseline CNN", "ResNet50+LoRA"],
            "Test Accuracy (%)": [19.33, 20.33],
        }
    )
    ax = sns.barplot(data=data, x="Model", y="Test Accuracy (%)", palette=["#7aa6c2", "#4c9f70"])
    ax.bar_label(ax.containers[0], fmt="%.2f", padding=3)
    ax.set_title("Active Learning: Test Accuracy Comparison")
    savefig("performance_comparison.png")


def ablation_results():
    # From COMPREHENSIVE_RESEARCH_REPORT.md (Task 3 table)
    import pandas as pd

    methods = [
        "CLIP Zero-shot",
        "CLIP+LoRA",
        "CLIP+LoRA+Synthetic",
        "CLIP+LoRA+Synthetic+MultiScale",
    ]
    accuracy = [45.3, 71.6, 83.7, 90.5]
    params_pct = [0.0, 1.78, 1.78, 2.13]

    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    color_acc = "#4c9f70"
    color_param = "#c27a7a"

    ax1.plot(methods, accuracy, marker="o", color=color_acc, label="Accuracy (%)")
    ax1.set_ylabel("Accuracy (%)", color=color_acc)
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_xticklabels(methods, rotation=20, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(methods, params_pct, marker="s", linestyle="--", color=color_param, label="Trainable Params (%)")
    ax2.set_ylabel("Trainable Params (%)", color=color_param)
    ax2.tick_params(axis="y", labelcolor=color_param)

    plt.title("Ablation: Accuracy vs Trainable Parameters")
    savefig("ablation_accuracy_params.png")


def parameter_efficiency():
    # Visualize trainable vs frozen parameters for Enhanced model
    # From enhanced model: total ~53.0M, trainable ~29.53M
    total = 53.041224
    trainable = 29.533192
    frozen = total - trainable

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.pie(
        [trainable, frozen],
        labels=[f"Trainable\n{trainable:,.1f}M", f"Frozen\n{frozen:,.1f}M"],
        colors=["#c27a7a", "#7aa6c2"],
        autopct="%1.1f%%",
        startangle=140,
        textprops={"fontsize": 9},
    )
    ax.set_title("Parameter Distribution (Enhanced Model)")
    savefig("parameter_efficiency_pie.png")


def active_learning_progression():
    # From ENHANCED_ACTIVE_LEARNING_RESULTS.md (round-by-round best val acc)
    rounds = list(range(1, 11))
    best_val = [22.67, 25.67, 23.33, 24.33, 24.33, 22.67, 24.67, 22.67, 24.67, 25.67]
    labeled = [100, 150, 198, 248, 297, 347, 397, 447, 496, 546]

    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot(rounds, best_val, marker="o", color="#4c9f70")
    ax1.set_xlabel("Active Learning Round")
    ax1.set_ylabel("Best Val Acc (%)", color="#4c9f70")
    ax1.tick_params(axis="y", labelcolor="#4c9f70")

    ax2 = ax1.twinx()
    ax2.plot(rounds, labeled, marker="s", linestyle="--", color="#7aa6c2")
    ax2.set_ylabel("Labeled Samples", color="#7aa6c2")
    ax2.tick_params(axis="y", labelcolor="#7aa6c2")

    plt.title("Active Learning Progression")
    savefig("active_learning_progression.png")


if __name__ == "__main__":
    performance_comparison()
    ablation_results()
    parameter_efficiency()
    active_learning_progression()

