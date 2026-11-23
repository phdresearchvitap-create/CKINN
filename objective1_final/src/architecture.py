from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def draw(path):
    fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
    ax.set(xlim=(0, 12), ylim=(0, 7))
    ax.axis("off")
    ink = "#103341"

    def box(x, y, w, h, title, sub, fill):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.05,rounding_size=0.1",
                facecolor=fill,
                edgecolor="#71969a",
                linewidth=1,
            )
        )
        ax.text(
            x + w / 2,
            y + h * 0.68,
            title,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=ink,
        )
        ax.text(x + w / 2, y + h * 0.29, sub, ha="center", va="center", fontsize=10, color=ink)

    def line(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", color="#4a7277", mutation_scale=14, lw=1.2))

    ax.text(0.25, 6.72, "Predicting recorded diabetes at three years", fontsize=21, weight="bold", color=ink)
    ax.text(
        0.25,
        6.3,
        "Baseline inputs use fold-fitted medians, positive scales and missingness indicators.",
        fontsize=12,
        color=ink,
    )
    for x, title, sub, n in [
        (0.25, "Demographics", "Age · sex", 4),
        (4.25, "Body and pressure", "BMI · systolic BP", 4),
        (8.25, "Laboratory", "ALT · FPG · TG · HDL-C", 8),
    ]:
        box(x, 4.96, 3.48, 0.95, title, sub, "#e6f2f0")
        ax.text(x + 1.74, 4.71, f"{n} inputs → Linear → SiLU → 8 values", ha="center", fontsize=10, color=ink)
        line((x + 1.74, 4.54), (6, 4.02))
    box(
        2.12,
        3.05,
        7.76,
        0.94,
        "Concatenate 24 learned values",
        "Linear 24 → 32 · SiLU · dropout · Linear 32 → 16 · SiLU",
        "#edf2f6",
    )
    line((6, 3.0), (6, 2.67))
    box(3.85, 1.76, 4.3, 0.87, "Linear 16 → 1 logit", "Sigmoid gives a three-year probability", "#cfe9e2")
    line((4.7, 1.71), (3.0, 1.21))
    line((7.3, 1.71), (9.0, 1.21))
    box(
        0.52,
        0.15,
        5.05,
        1.03,
        "IPCW outcome loss",
        "Weighted BCE with logits, averaged over all rows",
        "#f4f2e9",
    )
    box(
        6.42,
        0.15,
        5.05,
        1.03,
        "Soft clinical penalty",
        "Wrong-way probability gradients for observed FPG and BMI",
        "#f4f2e9",
    )
    ax.text(11.75, 6.72, "DEFAULT  ·  1,497 PARAMETERS", ha="right", fontsize=9, color="#487075")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=155, bbox_inches="tight", facecolor="white")
    plt.close(fig)
