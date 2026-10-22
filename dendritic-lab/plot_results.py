"""Generate paper figures (v2): two-panel design, tight axes, no legend overlap."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.6,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2, "ytick.major.size": 2})
W = 3.5

def two_panel(names, bwt, bwt_e, acc, acc_e, title, fname,
              bwt_lim=None, acc_lim=None, colors=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W, 2.7), sharex=True,
                                   gridspec_kw={"hspace": 0.12})
    x = np.arange(len(names))
    c = colors if colors else ["#888888"] * len(names)
    ax1.bar(x, bwt, 0.58, yerr=bwt_e, capsize=1.6, color=c)
    ax2.bar(x, acc, 0.58, yerr=acc_e, capsize=1.6, color=c)
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_ylabel("BWT", fontsize=7)
    ax2.set_ylabel("Acc (%)", fontsize=7)
    if bwt_lim: ax1.set_ylim(*bwt_lim)
    if acc_lim: ax2.set_ylim(*acc_lim)
    ax1.set_title(title, fontsize=7)
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=18, ha="right", fontsize=6.5)
    for a in (ax1, ax2):
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    fig.align_ylabels([ax1, ax2])
    fig.tight_layout(pad=0.25)
    fig.savefig(fname, dpi=300)
    plt.close(fig)

def fig_main():
    e12 = json.load(open("exp12_result.json")); e9 = json.load(open("exp9_result.json"))
    d = [("MLP", e9["mlp"]), ("AD", e9["ad"]), ("v1", e9["v1"]),
         ("SI", e12["5tasks"]["si"]), ("MAS", e12["5tasks"]["mas"]),
         ("PackNet", e12["5tasks"]["packnet"]), ("DeLP", e12["5tasks"]["ours"])]
    names = [k for k, _ in d]
    cols = ["#9a9a9a"] * 3 + ["#4C9A6B"] * 3 + ["#C0392B"]
    two_panel(names,
              [v["bwt_mean"] for _, v in d], [v["bwt_std"] for _, v in d],
              [v["final_avg_mean"] for _, v in d], [v["final_avg_std"] for _, v in d],
              "5-task Permuted-MNIST (5 seeds)", "fig_main_5t.png",
              bwt_lim=(-21, 1.5), acc_lim=(35, 86), colors=cols)

def fig_ablation():
    e4 = json.load(open("exp4_result.json")); e5b = json.load(open("exp4b_result.json"))
    e5c = json.load(open("exp4c_result.json")); e12 = json.load(open("exp12_result.json"))
    d = [("MLP", e4["mlp"]), ("AD", e4["ad"]), ("v1", e4["ad_prot"]),
         ("v2", e5b["ad_prot2"]), ("v3", e5c["ad_prot3"]),
         ("DeLP", e12["5tasks"]["ours"])]
    names = [k for k, _ in d]
    cols = ["#9a9a9a"] * 5 + ["#C0392B"]
    two_panel(names,
              [v["bwt_mean"] for _, v in d], [v["bwt_std"] for _, v in d],
              [v["final_avg_mean"] for _, v in d], [v["final_avg_std"] for _, v in d],
              "Ablation: isolating the interference channels", "fig_ablation.png",
              bwt_lim=(-23, 1.5), acc_lim=(20, 86), colors=cols)

def fig_10t():
    e12 = json.load(open("exp12_result.json")); e11 = json.load(open("exp11_result.json"))
    e8 = json.load(open("exp8_result_seeded.json"))
    d = [("SI", e12["10tasks"]["si"]), ("MAS", e12["10tasks"]["mas"]),
         ("PackNet", e12["10tasks"]["packnet"]), ("DeLP-5k", e12["10tasks"]["ours"]),
         ("DeLP-10k", e11["arms"]["lam10000"]), ("DeLP-last5", e11["arms"]["lam10000_M5"]),
         ("DeLP-pool", e8["10T_shared_oracle"])]
    names = [k for k, _ in d]
    cols = ["#4C9A6B"] * 3 + ["#C0392B"] * 4
    two_panel(names,
              [v["bwt_mean"] for _, v in d], [v["bwt_std"] for _, v in d],
              [v["final_avg_mean"] for _, v in d], [v["final_avg_std"] for _, v in d],
              "10-task stress test: the loss is in learning, not retention", "fig_10t.png",
              bwt_lim=(-24, 2), acc_lim=(15, 75), colors=cols)

if __name__ == "__main__":
    fig_main(); fig_ablation(); fig_10t()
    print("FIGURES_OK", sorted(f for f in os.listdir(".") if f.startswith("fig_")))
