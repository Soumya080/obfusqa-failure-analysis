"""Figure 1: share of obfuscation-induced failures by cause, for NEI / DI / CO."""
import matplotlib.pyplot as plt

from .attribute import ORDER

COLORS = {"scoring": "#9e9e9e", "distractor": "#e07b39", "resolution": "#3b75af", "selection": "#8c6bb1",
          "ambiguity": "#c44e52", "composition": "#2a9d8f", "unexplained": "#cccccc"}


def attribution_figure(summary, path, model_name=""):
    conds = [c for c in ["NEI", "DI", "CO"] if c in set(summary["condition"])]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for yi, cond in enumerate(conds):
        g = summary[summary["condition"] == cond].set_index("cause")
        left = 0.0
        n = int(g["n"].iloc[0]) if len(g) else 0
        for cause in ORDER:
            if cause not in g.index:
                continue
            share = float(g.loc[cause, "share"])
            if share <= 0:
                continue
            ax.barh(yi, share, left=left, color=COLORS[cause], edgecolor="white",
                    label=cause if yi == 0 or cause not in ax.get_legend_handles_labels()[1] else None)
            if share >= 0.07:
                ax.text(left + share / 2, yi, f"{share:.0%}", ha="center", va="center", fontsize=8,
                        color="#222222" if cause == "unexplained" else "white")
            left += share
        ax.text(1.01, yi, f"n={n}", va="center", fontsize=8)
    ax.set_yticks(range(len(conds)), conds)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of obfuscation-induced failures")
    ax.set_title(f"Why does the model fail? {model_name}".strip(), fontsize=10)
    ax.legend(ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.32), loc="upper center")
    ax.invert_yaxis()
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"[figure] saved {path}")
    return fig
