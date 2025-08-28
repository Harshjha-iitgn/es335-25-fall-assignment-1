"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
from tree.base import DecisionTree
from tree.utils import one_hot_encoding
from metrics import accuracy, mse, rmse, mae, r2, precision_recall_per_class, precision_for_class, recall_for_class

np.random.seed(42)

# ----------------------------
# Test case 1: Real Input, Real Output
# ----------------------------
print("\n=== Test 1: Real Input, Real Output ===")
N, P = 30, 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria)  # criterion is ignored for regression in our implementation
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print("Criteria:", criteria)
    print("MSE :", f"{mse(y, y_hat):.6f}")
    print("RMSE:", f"{rmse(y, y_hat):.6f}")
    print("MAE :", f"{mae(y, y_hat):.6f}")
    print("R2  :", f"{r2(y, y_hat):.6f}")
    print("-" * 40)

# ----------------------------
# Test case 2: Real Input, Discrete Output
# ----------------------------
print("\n=== Test 2: Real Input, Discrete Output ===")
N, P = 30, 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size=N), dtype="category")

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria)
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    acc = accuracy(y, y_hat)
    prec_pc, rec_pc, prec_macro, rec_macro = precision_recall_per_class(y, y_hat)
    print("Criteria:", criteria)
    print("Accuracy      :", f"{acc:.6f}")
    print("Macro Precision:", f"{prec_macro:.6f}", " Macro Recall:", f"{rec_macro:.6f}")
    for cls in y.unique():
        print(f"  Class {cls}: Precision {prec_pc[cls]:.6f} | Recall {rec_pc[cls]:.6f}")
    print("-" * 40)

# ----------------------------
# Test case 3: Discrete Input, Discrete Output
# ----------------------------
print("\n=== Test 3: Discrete Input, Discrete Output ===")
N, P = 30, 5
X_raw = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(P)})
X = one_hot_encoding(X_raw)  # convert to real-valued inputs
y = pd.Series(np.random.randint(P, size=N), dtype="category")

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria)
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    acc = accuracy(y, y_hat)
    prec_pc, rec_pc, prec_macro, rec_macro = precision_recall_per_class(y, y_hat)
    print("Criteria:", criteria)
    print("Accuracy      :", f"{acc:.6f}")
    print("Macro Precision:", f"{prec_macro:.6f}", " Macro Recall:", f"{rec_macro:.6f}")
    for cls in y.unique():
        print(f"  Class {cls}: Precision {prec_pc[cls]:.6f} | Recall {rec_pc[cls]:.6f}")
    print("-" * 40)

# ----------------------------
# Test case 4: Discrete Input, Real Output
# ----------------------------
print("\n=== Test 4: Discrete Input, Real Output ===")
N, P = 30, 5
X_raw = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(P)})
X = one_hot_encoding(X_raw)  # convert to real-valued inputs
y = pd.Series(np.random.randn(N))

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria)  # ignored for regression
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print("Criteria:", criteria)
    print("MSE :", f"{mse(y, y_hat):.6f}")
    print("RMSE:", f"{rmse(y, y_hat):.6f}")
    print("MAE :", f"{mae(y, y_hat):.6f}")
    print("R2  :", f"{r2(y, y_hat):.6f}")
    print("-" * 40)
