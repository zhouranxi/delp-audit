"""Fig.1: conceptual diagram -- active-dendrite neuron (a) and the three
interference channels with their measured consequences (b)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

plt.rcParams.update({"font.size": 7, "font.family": "serif", "mathtext.fontset": "dejavuserif",
                     "axes.linewidth": 0.6})

GREY, DARK, RED, AMBER = "#9a9a9a", "#2c2c2c", "#C0392B", "#d99b2b"


def box(ax, x, y, w, h, label, fc="#f2f2f2", ec="#444444", fs=6.6, lw=0.7, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs,
            color="#1a1a1a", zorder=4, fontweight="bold" if bold else "normal")


def arrow(ax, p0, p1, color="#444444", lw=0.7, style="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=5, lw=lw,
                                 color=color, zorder=2, shrinkA=0, shrinkB=0))


def panel_a(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.0, 1.05, "(a) Active-dendrite neuron", fontsize=7,
            ha="left", va="bottom", fontweight="bold")

    # context branch (top)
    box(ax, 0.020, 0.735, 0.070, 0.140, r"$\mathbf{c}$", fs=7.5)
    box(ax, 0.125, 0.735, 0.058, 0.140, "$U$")
    arrow(ax, (0.090, 0.805), (0.125, 0.805))

    seg_x, seg_w, seg_h = 0.240, 0.120, 0.062
    ys = [0.617, 0.690, 0.763, 0.836]
    for i, y0 in enumerate(ys):
        win = (i == 1)
        box(ax, seg_x, y0, seg_w, seg_h, f"seg {i+1}",
            fc="#f0c9c3" if win else "#dcdcdc",
            ec=RED if win else "#777", fs=6.0,
            lw=1.1 if win else 0.7, bold=win)
        arrow(ax, (0.183, 0.805), (seg_x - 0.004, y0 + seg_h / 2), color="#777", lw=0.5)
        arrow(ax, (seg_x + seg_w, y0 + seg_h / 2), (0.398, 0.700), color="#777", lw=0.5)

    box(ax, 0.398, 0.664, 0.052, 0.072, "max", fs=6.0)
    box(ax, 0.472, 0.664, 0.050, 0.072, r"$\sigma$", fs=7.2)
    arrow(ax, (0.450, 0.700), (0.472, 0.700))

    # feedforward path (bottom)
    ax.add_patch(Circle((0.030, 0.220), 0.026, fc="#e8e8e8", ec="#444", lw=0.7, zorder=3))
    ax.text(0.030, 0.220, r"$\mathbf{x}$", ha="center", va="center", fontsize=7, zorder=4)
    box(ax, 0.080, 0.135, 0.115, 0.170, "$W_{ff}$\nshared", fc="#efcfc9", ec=RED, fs=6.4)
    ax.add_patch(Circle((0.248, 0.220), 0.023, fc="white", ec="#444", lw=0.7, zorder=3))
    ax.text(0.248, 0.220, r"$\times$", ha="center", va="center", fontsize=7.5, zorder=4)
    box(ax, 0.298, 0.135, 0.095, 0.170, "k-WTA")
    box(ax, 0.425, 0.135, 0.105, 0.170, "$W_{out}$\nreadout", fc="#f6e3c4", ec=AMBER, fs=6.4)
    ax.add_patch(Circle((0.590, 0.220), 0.026, fc="#e8e8e8", ec="#444", lw=0.7, zorder=3))
    ax.text(0.590, 0.220, r"$\hat{y}$", ha="center", va="center", fontsize=7, zorder=4)
    for x0, x1 in [(0.056, 0.080), (0.195, 0.225), (0.271, 0.298), (0.393, 0.425), (0.530, 0.564)]:
        arrow(ax, (x0, 0.220), (x1, 0.220))

    # gating path: sigma -> down -> left -> multiply
    arrow(ax, (0.497, 0.664), (0.497, 0.430), color=RED, lw=0.8, style="-")
    arrow(ax, (0.497, 0.430), (0.248, 0.430), color=RED, lw=0.8, style="-")
    arrow(ax, (0.248, 0.430), (0.248, 0.243), color=RED, lw=0.8)
    ax.text(0.372, 0.442, "gating signal", fontsize=5.9, color=RED, ha="center", va="bottom")

    ax.text(0.020, 0.900, r"context $\mathbf{c}$", fontsize=6.0, color="#555",
            ha="left", va="center", style="italic")
    ax.text(0.020, 0.075, "The winning segment (red) gates the shared drive; non-current\n"
                          "segments are frozen while a new task is learned.",
            fontsize=5.9, color="#333", ha="left", va="center")


def panel_b(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.0, 1.05, "(b) Where the interference lives", fontsize=7,
            ha="left", va="bottom", fontweight="bold")

    rows = [
        (0.600, "$U$", "segments", "#ececec", "#777",
         "freeze only (v1)", r"BWT $-14.8$", "no effect", GREY),
        (0.330, "$W_{ff}$", "shared", "#efcfc9", RED,
         "usage-weighted (DeLP)", r"BWT $-2.2$", "dominant", RED),
        (0.060, "$W_{out}$", "readout", "#f6e3c4", AMBER,
         "per-task head (v3)", r"BWT $-19.8$", "relocates", AMBER),
    ]
    for y, sym, sub, fc, ec, intervention, bwt, verdict, vcol in rows:
        ax.add_patch(FancyBboxPatch((0.005, y), 0.150, 0.175,
                                    boxstyle="round,pad=0.004,rounding_size=0.012",
                                    fc=fc, ec=ec, lw=0.9, zorder=3))
        ax.text(0.080, y + 0.112, sym, ha="center", va="center", fontsize=7.0, zorder=4)
        ax.text(0.080, y + 0.052, sub, ha="center", va="center", fontsize=5.9, zorder=4)
        ax.text(0.175, y + 0.108, intervention, ha="left", va="center", fontsize=6.0,
                color="#222")
        ax.text(0.175, y + 0.048, bwt, ha="left", va="center", fontsize=6.4,
                color=vcol, fontweight="bold")
        ax.text(0.997, y + 0.080, verdict, ha="right", va="center", fontsize=6.0,
                color=vcol, style="italic")


def main():
    fig = plt.figure(figsize=(7.0, 2.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.07)
    panel_a(fig.add_subplot(gs[0, 0]))
    panel_b(fig.add_subplot(gs[0, 1]))
    fig.subplots_adjust(left=0.025, right=0.985, top=0.90, bottom=0.02)
    fig.savefig("fig_concept.png", dpi=300)
    plt.close(fig)
    print("FIG_CONCEPT_OK")


if __name__ == "__main__":
    main()
