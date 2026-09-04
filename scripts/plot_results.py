"""Plot three baseline benchmarks plus protoGDA drug/scaffold consistency.

Data source: Tables 1 and 2 in docs/paper_draft.md.
Output: docs/figures/results_bar.png (Figure 5).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "docs", "figures", "results_bar.png")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "figure.facecolor": "#FAF9F7",
        "axes.facecolor": "#FAF9F7",
        "axes.edgecolor": "#5B5A57",
        "axes.labelcolor": "#32312F",
        "xtick.color": "#32312F",
        "ytick.color": "#32312F",
        "text.color": "#32312F",
    }
)
plt.rcParams["axes.unicode_minus"] = False

MODELS = ["CANDELA", "MGATAF", "protoGDA"]
COLORS = {
    "CANDELA": "#8D9EAA",  # muted slate blue
    "MGATAF": "#C4A06A",  # muted ochre
    "protoGDA": "#6E8F8B",  # muted sage teal
}

# Baseline comparison settings -> {model: (mean CV Pearson, standard deviation)}
DATA = {
    "Interpolation": {
        "CANDELA": (0.8945, 0.0039),
        "MGATAF": (0.8278, 0.0024),
        "protoGDA": (0.9344, 0.0016),
    },
    "Cell Cold-Start": {
        "CANDELA": (0.8372, 0.0209),
        "MGATAF": (0.8152, 0.0053),
        "protoGDA": (0.8735, 0.0036),
    },
    "Drug Cold-Start": {
        "CANDELA": (0.3370, 0.1089),
        "MGATAF": (0.4325, 0.1603),
        "protoGDA": (0.5305, 0.1842),
    },
}

# Deterministic protoGDA protocol consistency results (Table 2).
PROTOCOL_DATA = {
    "Drug Cold-Start": (0.5305, 0.1842),
    "Scaffold Cold-Start": (0.5067, 0.1035),
}

BAR_W = 0.5
YLIM = (0.0, 1.05)


def _annotate(ax, x, y, text, color, dy=0.0):
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(0, dy),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=color,
    )


def main():
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.25))
    axes = axes.ravel()

    for idx, scene in enumerate(DATA):
        ax = axes[idx]
        values = DATA[scene]
        x = np.arange(len(MODELS))
        means = [values[m][0] for m in MODELS]
        stds = [values[m][1] for m in MODELS]
        bars = ax.bar(
            x,
            means,
            BAR_W,
            yerr=stds,
            capsize=3,
            error_kw={"elinewidth": 1.1, "ecolor": "#535452", "capthick": 1.1},
            color=[COLORS[m] for m in MODELS],
            edgecolor="#FAF9F7",
            linewidth=0.8,
            zorder=3,
        )

        for xi, (m, bar) in enumerate(zip(MODELS, bars)):
            top = max(means[xi], means[xi] + stds[xi])
            _annotate(ax, xi, top + 0.015, f"{means[xi]:.3f}", COLORS[m], dy=0)

        ax.set_ylim(*YLIM)
        ax.set_xlim(-0.48, 2.58)
        ax.set_title(scene, fontsize=12, fontweight="bold", pad=9)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, fontsize=10)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", color="#D9D6D0", linestyle=(0, (2, 2)), linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # Fourth panel: within-model comparison rather than a cross-model ranking.
    ax = axes[3]
    labels = list(PROTOCOL_DATA)
    means = [PROTOCOL_DATA[label][0] for label in labels]
    stds = [PROTOCOL_DATA[label][1] for label in labels]
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        means,
        BAR_W,
        yerr=stds,
        capsize=3,
        error_kw={"elinewidth": 1.1, "ecolor": "#535452", "capthick": 1.1},
        color="#6E8F8B",
        edgecolor="#FAF9F7",
        linewidth=0.8,
        zorder=3,
    )
    for xi, bar in enumerate(bars):
        _annotate(ax, xi, means[xi] + stds[xi] + 0.015, f"{means[xi]:.3f}", "#6E8F8B")
    ax.set_ylim(*YLIM)
    ax.set_xlim(-0.55, 1.55)
    ax.set_title("protoGDA: Protocol Consistency", fontsize=12, fontweight="bold", pad=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["Drug Cold-Start", "Scaffold Cold-Start"], fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", color="#D9D6D0", linestyle=(0, (2, 2)), linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.text(0.5, 0.035, "CV Pearson Correlation", ha="center", va="center", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0), h_pad=2.0, w_pad=1.2)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
