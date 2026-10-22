"""gen_jobs_d2.py -- job list for the dual-compartment study.

phases
  curve  : 8 arms x 6 gaps x 3 seeds        -> the accuracy-vs-gap family
  ktau   : kappa x tau_a grid at two gaps   -> mechanism surface
  taulaw : gap sweep per tau_a              -> the Delta_1/2 = c * t_half law
"""
import argparse

COMBOS = [("point", "thr"), ("point_slow", "thr"), ("point_gate", "thr"),
          ("alif", "thr"), ("dual", "add"), ("dual", "mul"), ("dual", "thr"),
          ("dual_alif", "thr")]
GAPS = [2, 5, 10, 20, 40, 80]
TAUS = [0.90, 0.95, 0.97, 0.99]
KAPPAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
COMMON = "--epochs 20 --nh 96 --ntrain 4000 --ntest 1000"


def emit(jobs):
    for i, (arm, couple, gap, kappa, tau_a, seed) in enumerate(jobs):
        tag = f"{arm}_{couple}_g{gap}_k{kappa:g}_ta{tau_a:g}_s{seed}"
        print(f"{i % 4}\t{tag}\td2_comp_snn.py --arm {arm} --couple {couple} "
              f"--gap {gap} --kappa {kappa:g} --tau_a {tau_a:g} --seed {seed} "
              f"{COMMON} --out results_d2/{tag}.json")
    print(f"# {len(jobs)} jobs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="curve")
    a = ap.parse_args()
    jobs = []
    if a.phase == "curve":
        for arm, couple in COMBOS:
            for gap in GAPS:
                for seed in (0, 1, 2):
                    jobs.append((arm, couple, gap, 0.8, 0.97, seed))
    elif a.phase == "ktau":
        for kappa in KAPPAS:
            for tau_a in TAUS:
                for gap in (5, 20):
                    jobs.append(("dual", "thr", gap, kappa, tau_a, 0))
    elif a.phase == "taulaw":
        for tau_a in TAUS:
            for gap in GAPS:
                for seed in (0, 1):
                    jobs.append(("dual", "thr", gap, 0.8, tau_a, seed))
    emit(jobs)


if __name__ == "__main__":
    main()
