import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.model_selection import train_test_split
from metrics import *
from sklearn.datasets import make_classification

# ========================= Q2 (a): 70/30 split + metrics =========================
# Wrap into DataFrame/Series for consistency with our tree API
# dataset given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2,
    random_state=1, n_clusters_per_class=2, class_sep=0.5
)

X_df = pd.DataFrame(X, columns=["x1", "x2"])
y_s  = pd.Series(y)

# 70/30 split (shuffle once with a fixed seed)
rng = np.random.default_rng(1)
idx = rng.permutation(len(y_s))
cut = int(0.7 * len(y_s))
tr_idx, te_idx = idx[:cut], idx[cut:]

Xtr, Xte = X_df.iloc[tr_idx], X_df.iloc[te_idx]
ytr, yte = y_s.iloc[tr_idx], y_s.iloc[te_idx]

# Train a tree (classification). You can switch to "gini_index" if you prefer.
dt = DecisionTree(criterion="information_gain", max_depth=5)
dt.fit(Xtr, ytr)
yp = dt.predict(Xte)

# Metrics: accuracy + per-class precision/recall (macro included)
acc = accuracy(yte, yp)
prec_pc, rec_pc, prec_macro, rec_macro = precision_recall_per_class(yte, yp)

print("\n=== Q2 (a): 70/30 Train/Test ===")
print(f"Accuracy        : {acc:.4f}")
print(f"Macro Precision : {prec_macro:.4f}")
print(f"Macro Recall    : {rec_macro:.4f}")
for cls in np.unique(y):
    print(f"  Class {cls}: Precision {prec_pc[cls]:.4f} | Recall {rec_pc[cls]:.4f}")

# Optional: visualize decision regions for the trained tree
x_min, x_max = X_df["x1"].min() - 1.0, X_df["x1"].max() + 1.0
y_min, y_max = X_df["x2"].min() - 1.0, X_df["x2"].max() + 1.0
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                     np.linspace(y_min, y_max, 400))
grid = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=["x1", "x2"])
zz = dt.predict(grid).to_numpy().reshape(xx.shape)

plt.figure()
plt.contourf(xx, yy, zz, alpha=0.25, cmap='bwr')
plt.scatter(Xtr["x1"], Xtr["x2"], c=ytr, cmap='bwr', edgecolors='k', s=20, label="train")
plt.scatter(Xte["x1"], Xte["x2"], c=yte, cmap='bwr', marker='x', s=40, label="test")
plt.title("(a) Decision regions (max_depth=5, entropy)")
plt.xlabel("x1"); plt.ylabel("x2"); plt.legend(); plt.tight_layout()
plt.show()

# ========================= Q2 (b): 5-fold nested cross-validation =========================
def kfold_indices(n, k, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return np.array_split(idx, k)

def nested_cv_best_depth(X_df, y_s, depths=(1,2,3,4,5,6,8,10), outer_k=5, inner_k=5,
                         seed_outer=42, seed_inner=123):
    n = len(y_s)
    outer_folds = kfold_indices(n, outer_k, seed_outer)
    best_depths, outer_scores = [], []

    for outer_test_idx in outer_folds:
        outer_train_idx = np.setdiff1d(np.arange(n), outer_test_idx)
        Xtr, ytr = X_df.iloc[outer_train_idx], y_s.iloc[outer_train_idx]
        Xte, yte = X_df.iloc[outer_test_idx], y_s.iloc[outer_test_idx]

        # Inner CV to pick best depth
        inner_folds = kfold_indices(len(ytr), inner_k, seed_inner)
        cv_scores = []
        for d in depths:
            fold_scores = []
            for inner_val_idx in inner_folds:
                inner_train_idx = np.setdiff1d(np.arange(len(ytr)), inner_val_idx)
                Xtr_tr, ytr_tr = Xtr.iloc[inner_train_idx], ytr.iloc[inner_train_idx]
                Xtr_val, ytr_val = Xtr.iloc[inner_val_idx], ytr.iloc[inner_val_idx]

                dt_inner = DecisionTree(criterion="information_gain", max_depth=d)
                dt_inner.fit(Xtr_tr, ytr_tr)
                yp_val = dt_inner.predict(Xtr_val)
                fold_scores.append(accuracy(ytr_val, yp_val))
            cv_scores.append((d, float(np.mean(fold_scores))))

        # pick depth with highest inner-CV accuracy
        d_star = max(cv_scores, key=lambda t: t[1])[0]
        best_depths.append(d_star)

        # retrain on all outer-train with d_star, evaluate on outer-test
        dt_outer = DecisionTree(criterion="information_gain", max_depth=d_star)
        dt_outer.fit(Xtr, ytr)
        yp_outer = dt_outer.predict(Xte)
        outer_scores.append(accuracy(yte, yp_outer))

    return best_depths, outer_scores

depths_to_try = (1,2,3,4,5,6,8,10)
best_depths, outer_scores = nested_cv_best_depth(X_df, y_s, depths=depths_to_try)

print("\n=== Q2 (b): Nested CV (5x5) ===")
print("Best depth per outer fold:", best_depths)
print("Outer-fold accuracies    :", [f"{s:.4f}" for s in outer_scores])
print("Mean outer accuracy      :", f"{np.mean(outer_scores):.4f}")

# Optional: bar plot of outer scores
plt.figure()
plt.bar(range(1, len(outer_scores)+1), outer_scores)
plt.xticks(range(1, len(outer_scores)+1))
plt.ylim(0, 1)
plt.xlabel("Outer fold"); plt.ylabel("Accuracy")
plt.title("(b) Outer-fold accuracies (depth from inner CV)")
plt.tight_layout()
plt.show()
