#!/usr/bin/env python3
"""
Generate a poster-style summary figure combining key results and diagrams.
Outputs: docs/figures/poster_summary.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

FIG_DIR = Path("docs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    fig = plt.figure(figsize=(11, 8))  # landscape
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.2], height_ratios=[1, 1], wspace=0.25, hspace=0.25)

    # Title
    fig.suptitle("Adaptive Foundation Models for PCB Defect Detection", fontsize=16, fontweight="bold")

    # Panels
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    def try_imshow(ax, rel_path, title):
        p = FIG_DIR / rel_path
        if p.exists():
            ax.imshow(mpimg.imread(p))
            ax.set_title(title)
            ax.axis("off")
        else:
            ax.text(0.5, 0.5, f"Missing:\n{rel_path}", ha="center", va="center")
            ax.set_title(title)
            ax.axis("off")

    # Top row: performance, ablation, parameter efficiency
    try_imshow(axes[0], "performance_comparison.png", "Active Learning Performance")
    try_imshow(axes[1], "ablation_accuracy_params.png", "Ablation: Acc vs Params")
    try_imshow(axes[2], "parameter_efficiency_pie.png", "Parameter Distribution")

    # Bottom row: active learning progression, GradCAM (if exists), placeholder architecture
    try_imshow(axes[3], "active_learning_progression.png", "Active Learning Progression")

    # GradCAM from outputs if available, else leave placeholder
    gradcam_path = Path("outputs/explainability/gradcam_analysis.png")
    if gradcam_path.exists():
        axes[4].imshow(mpimg.imread(gradcam_path))
        axes[4].set_title("Explainability: Grad-CAM")
        axes[4].axis("off")
    else:
        axes[4].text(0.5, 0.5, "Run run_gradcam.py\nto include Grad-CAM", ha="center", va="center")
        axes[4].set_title("Explainability: Grad-CAM")
        axes[4].axis("off")

    # Architecture placeholder with text (Mermaid not rendered here)
    axes[5].text(
        0.05,
        0.5,
        "System Architecture:\n"
        "Data → Preprocess → Backbone →\n"
        "Pyramid Attention → LoRA → Head\n"
        "↻ Active Learning Loop",
        fontsize=11,
        va="center",
    )
    axes[5].axis("off")

    out = FIG_DIR / "poster_summary.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

