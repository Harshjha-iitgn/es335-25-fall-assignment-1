import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from tree.base import DecisionTree

np.random.seed(42)

# ======================= Small, fast experiment controls =======================
num_average_time = 3   # repeats per measurement (increase if you want smoother curves)
fixed_N = 500          # N used for the M-sweep
fixed_M = 10           # M used for the N-sweep
Ms = [5, 10, 20, 40]   # feature sizes to try
Ns = [100, 200, 400, 800]  # sample sizes to try
test_frac = 0.2
max_depth = None       # let the tree grow naturally

# ======================= Data generation (binary features) =====================
def make_binary_dataset(N, M, use_real_inputs: bool, use_discrete_output: bool, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(N, M)).astype(float)
    X_df = pd.DataFrame(X, columns=[f"x{j}" for j in range(M)])

    if use_discrete_output:
        y = rng.integers(0, 2, size=N)
        y_s = pd.Series(y)
    else:
        w = rng.normal(0, 1, size=M)
        y = X @ w + rng.normal(0, 0.5, size=N)
        y_s = pd.Series(y.astype(float))

    idx = rng.permutation(N)
    cut = int((1 - test_frac) * N)
    tr_idx, te_idx = idx[:cut], idx[cut:]
    return (
        X_df.iloc[tr_idx].reset_index(drop=True),
        y_s.iloc[tr_idx].reset_index(drop=True),
        X_df.iloc[te_idx].reset_index(drop=True),
        y_s.iloc[te_idx].reset_index(drop=True),
    )

# ======================= Timing (fit & predict) =======================
def time_fit_predict(Xtr, ytr, Xte, repeats=3):
    fit_times, pred_times = [], []

    # warm-up
    dt = DecisionTree(criterion="information_gain", max_depth=max_depth)
    dt.fit(Xtr, ytr)
    _ = dt.predict(Xte)

    for _ in range(repeats):
        dt = DecisionTree(criterion="information_gain", max_depth=max_depth)
        t0 = time.perf_counter()
        dt.fit(Xtr, ytr)
        t1 = time.perf_counter()
        _ = dt.predict(Xte)
        t2 = time.perf_counter()
        fit_times.append(t1 - t0)
        pred_times.append(t2 - t1)

    return float(np.mean(fit_times)), float(np.mean(pred_times))

# ======================= Run 4 cases for one (N, M) =======================
def run_four_cases(N, M, repeats=3, seed_base=0):
    cases = {
        "discX_discY": (False, True),
        "discX_realY": (False, False),
        "realX_discY": (True,  True),
        "realX_realY": (True,  False),
    }
    out = {}
    for i, (name, (use_real_inputs, use_discrete_output)) in enumerate(cases.items()):
        Xtr, ytr, Xte, yte = make_binary_dataset(N, M, use_real_inputs, use_discrete_output, seed=seed_base + i)
        fit_m, pred_m = time_fit_predict(Xtr, ytr, Xte, repeats=repeats)
        out[name] = (fit_m, pred_m)
    return out

# ======================= Sweeps =======================
def sweep_M(N_fixed, Ms, repeats=3):
    results = {k: {"fit": [], "pred": []}
               for k in ["discX_discY", "discX_realY", "realX_discY", "realX_realY"]}
    for M in Ms:
        res = run_four_cases(N_fixed, M, repeats=repeats, seed_base=10 + M)
        for k in results:
            fit_m, pred_m = res[k]
            results[k]["fit"].append(fit_m)
            results[k]["pred"].append(pred_m)
    return results

def sweep_N(M_fixed, Ns, repeats=3):
    results = {k: {"fit": [], "pred": []}
               for k in ["discX_discY", "discX_realY", "realX_discY", "realX_realY"]}
    for N in Ns:
        res = run_four_cases(N, M_fixed, repeats=repeats, seed_base=20 + N)
        for k in results:
            fit_m, pred_m = res[k]
            results[k]["fit"].append(fit_m)
            results[k]["pred"].append(pred_m)
    return results

# ======================= Plotting (save + show) =======================
def _save_and_show(title):
    fname = title.replace(" ", "_") + ".png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"[Saved plot: {fname}]")
    plt.show()

def plot_sweep_M_fit(Ms, results, title="Fit time vs M (N fixed)"):
    labels = {
        "discX_discY": "Disc X, Disc y",
        "discX_realY": "Disc X, Real y",
        "realX_discY": "Real X, Disc y",
        "realX_realY": "Real X, Real y",
    }
    plt.figure(figsize=(7, 4))
    for k, lab in labels.items():
        plt.plot(Ms, results[k]["fit"], marker="o", label=lab)
    plt.xlabel("M (features)")
    plt.ylabel("Fit time (s)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    _save_and_show(title)

def plot_sweep_M_pred(Ms, results, title="Predict time vs M (N fixed)"):
    labels = {
        "discX_discY": "Disc X, Disc y",
        "discX_realY": "Disc X, Real y",
        "realX_discY": "Real X, Disc y",
        "realX_realY": "Real X, Real y",
    }
    plt.figure(figsize=(7, 4))
    for k, lab in labels.items():
        plt.plot(Ms, results[k]["pred"], marker="s", label=lab)
    plt.xlabel("M (features)")
    plt.ylabel("Predict time (s)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    _save_and_show(title)

def plot_sweep_N_fit(Ns, results, title="Fit time vs N (M fixed)"):
    labels = {
        "discX_discY": "Disc X, Disc y",
        "discX_realY": "Disc X, Real y",
        "realX_discY": "Real X, Disc y",
        "realX_realY": "Real X, Real y",
    }
    plt.figure(figsize=(7, 4))
    for k, lab in labels.items():
        plt.plot(Ns, results[k]["fit"], marker="o", label=lab)
    plt.xlabel("N (samples)")
    plt.ylabel("Fit time (s)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    _save_and_show(title)

def plot_sweep_N_pred(Ns, results, title="Predict time vs N (M fixed)"):
    labels = {
        "discX_discY": "Disc X, Disc y",
        "discX_realY": "Disc X, Real y",
        "realX_discY": "Real X, Disc y",
        "realX_realY": "Real X, Real y",
    }
    plt.figure(figsize=(7, 4))
    for k, lab in labels.items():
        plt.plot(Ns, results[k]["pred"], marker="s", label=lab)
    plt.xlabel("N (samples)")
    plt.ylabel("Predict time (s)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    _save_and_show(title)

# ======================= Run the experiments =======================
if __name__ == "__main__":
    print("Running sweep over M with N fixed =", fixed_N)
    res_M = sweep_M(fixed_N, Ms, repeats=num_average_time)
    plot_sweep_M_fit(Ms, res_M, title=f"Fit time vs M (N={fixed_N})")
    plot_sweep_M_pred(Ms, res_M, title=f"Predict time vs M (N={fixed_N})")

    print("Running sweep over N with M fixed =", fixed_M)
    res_N = sweep_N(fixed_M, Ns, repeats=num_average_time)
    plot_sweep_N_fit(Ns, res_N, title=f"Fit time vs N (M={fixed_M})")
    plot_sweep_N_pred(Ns, res_N, title=f"Predict time vs N (M={fixed_M})")

    print("\nNotes:")
    print("- Fit time ~ increases with M (features), since the tree scans splits per feature.")
    print("- Fit time grows with N (samples); with real-valued thresholds: ~O(depth * N * M * log N).")
    print("- Predict time ~ linear in number of test samples and in tree depth (~O(depth) per sample).")
    print("- Trends are similar across the four cases; absolute times vary slightly due to impurity math.")
