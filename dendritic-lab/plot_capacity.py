"""Fig.4: learning/retention decomposition (a) and the stability-plasticity
frontier at ten tasks (b).  Reads exp10_result.json + exp11_result.json."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.6,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2, "ytick.major.size": 2})

GREY, RED, RED2, RED3, GREEN, DARK = "#8c8c8c", "#C0392B", "#e07b6b", "#f0b3a5", "#4C9A6B", "#333333"


def panel_a(ax, e10):
    runs = e10["runs"]
    x = np.arange(1, 11)
    series = [("MLP (N=256)", "mlp256", GREY, "-", "o"),
              ("DeLP N=256", "delp256", RED, "-", "s"),
              ("DeLP N=512", "delp512", RED2, "--", "^"),
              ("DeLP N=1024", "delp1024", RED3, "--", "D")]
    for lab, key, col, ls, mk in series:
        y = runs[key]["justlearn"]
        sd = runs[key]["justlearn_sd"]
        ax.errorbar(x, y, yerr=sd, color=col, ls=ls, marker=mk, ms=2.6, lw=0.9,
                    capsize=1.2, elinewidth=0.5, label=lab)

    ceil_delp = np.mean(list(e10["ceiling"]["delp256"].values()))
    ceil_mlp = np.mean(list(e10["ceiling"]["mlp256"].values()))
    ax.axhline(ceil_delp, color=RED, ls=":", lw=0.8)
    ax.axhline(ceil_mlp, color=GREY, ls=":", lw=0.8)
    ax.text(10.55, ceil_delp + 5.0, f"single-task\nceiling {ceil_delp:.0f}", fontsize=5.6,
            color=RED, va="center", ha="left")
    ax.text(10.55, ceil_mlp - 8.0, f"single-task\nceiling {ceil_mlp:.0f}", fontsize=5.6,
            color=GREY, va="center", ha="left")

    ax.set_xlim(0.5, 12.4); ax.set_ylim(10, 98)
    ax.set_xlabel("task index in the sequence", fontsize=6.6)
    ax.set_ylabel("accuracy just after learning (%)", fontsize=6.6)
    ax.set_title("(a) The loss is in learning, not in retention", fontsize=7, loc="left")
    ax.legend(fontsize=5.8, frameon=False, loc="lower left", handlelength=1.6,
              borderpad=0.2, labelspacing=0.25)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def panel_b(ax, e11):
    arms = e11["arms"]
    order = ["lam0", "lam100", "lam1000", "lam5000", "lam10000"]
    lab = {"lam0": r"$\lambda$=0", "lam100": r"$\lambda$=100",
           "lam1000": r"$\lambda$=1k", "lam5000": r"$\lambda$=5k",
           "lam10000": r"$\lambda$=10k"}
    xs = [arms[k]["bwt_mean"] for k in order]
    ys = [arms[k]["final_avg_mean"] for k in order]
    ax.plot(xs, ys, color=RED, lw=0.9, marker="o", ms=2.8, zorder=3,
            label="DeLP, all anchors")
    for k, xv, yv in zip(order, xs, ys):
        ax.annotate(lab[k], (xv, yv), textcoords="offset points", xytext=(3, 4),
                    fontsize=5.6, color=RED)

    mx = [arms[k]["bwt_mean"] for k in ("lam5000_M5", "lam10000_M5")]
    my = [arms[k]["final_avg_mean"] for k in ("lam5000_M5", "lam10000_M5")]
    ax.plot(mx, my, color=RED, lw=0.7, ls="--", marker="o", ms=3.2, mfc="white",
            mew=0.9, zorder=4, label="DeLP, 5 most recent anchors")
    ax.annotate("5k, M=5", (mx[0], my[0]), textcoords="offset points", xytext=(-26, -3),
                fontsize=5.6, color=RED)
    ax.annotate("10k, M=5", (mx[1], my[1]), textcoords="offset points", xytext=(4, 6),
                fontsize=5.6, color=RED)

    ref = e11["reference_10t"]
    ax.scatter([ref["si"]["bwt_mean"]], [ref["si"]["final_avg_mean"]], marker="*", s=42,
               color=GREEN, zorder=5, label="Synaptic Intelligence")
    ax.annotate("SI", (ref["si"]["bwt_mean"], ref["si"]["final_avg_mean"]),
                textcoords="offset points", xytext=(4, -7), fontsize=6.0, color=GREEN)
    ax.scatter([ref["mas"]["bwt_mean"]], [ref["mas"]["final_avg_mean"]], marker="s", s=9,
               color=GREEN, zorder=5, label="MAS")
    ax.annotate("MAS", (ref["mas"]["bwt_mean"], ref["mas"]["final_avg_mean"]),
                textcoords="offset points", xytext=(4, 3), fontsize=6.0, color=GREEN)

    ax.set_xlabel("BWT (higher = less forgetting)", fontsize=6.6)
    ax.set_ylabel("final average accuracy (%)", fontsize=6.6)
    ax.set_title("(b) Ten tasks: one frontier, two operating points", fontsize=7, loc="left")
    ax.set_xlim(-21.5, 0.0); ax.set_ylim(48, 79)
    ax.legend(fontsize=5.6, frameon=False, loc="lower left", handlelength=1.4,
              borderpad=0.2, labelspacing=0.25)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def main():
    e10 = json.load(open("exp10_result.json"))
    e11 = json.load(open("exp11_result.json"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.3),
                                 gridspec_kw={"wspace": 0.30})
    panel_a(a1, e10)
    panel_b(a2, e11)
    fig.subplots_adjust(left=0.065, right=0.90, top=0.90, bottom=0.145)
    fig.savefig("fig_capacity.png", dpi=300)
    plt.close(fig)
    print("FIG_CAPACITY_OK")


if __name__ == "__main__":
    main()
