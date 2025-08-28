import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tree.base import DecisionTree
from tree.utils import one_hot_encoding
from metrics import mse, rmse, mae, r2

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

# -------------------- Load & clean the Auto MPG dataset --------------------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
cols = ["mpg", "cylinders", "displacement", "horsepower", "weight",
        "acceleration", "model year", "origin", "car name"]

# 'horsepower' has '?' for missing — treat as NaN via na_values
data = pd.read_csv(
    url, delim_whitespace=True, header=None, names=cols, na_values=["?"]
)

# Drop rows with missing values we can't use
data = data.dropna().reset_index(drop=True)

# Target
y = data["mpg"]

# Features (drop non-numeric text column)
X = data.drop(columns=["mpg", "car name"])

# Treat some integer-coded columns as categorical (discrete) and one-hot them
for c in ["origin", "cylinders", "model year"]:
    X[c] = X[c].astype("category")

# Convert all categorical columns to 0/1 (floats) so our tree gets real inputs only
X = one_hot_encoding(X)  # returns all-float DataFrame

# -------------------- Train/Test split --------------------
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=42)

# ===================== (a) Your Decision Tree (Regression) =====================
# criterion in our implementation is ignored for regression; tree uses MSE reduction internally
dt = DecisionTree(criterion="information_gain", max_depth=None)
dt.fit(Xtr, ytr)
y_pred = dt.predict(Xte)

print("\n=== (a) Our DecisionTree on Auto MPG ===")
print(f"MSE : {mse(yte, y_pred):.4f}")
print(f"RMSE: {rmse(yte, y_pred):.4f}")
print(f"MAE : {mae(yte, y_pred):.4f}")
print(f"R^2 : {r2(yte, y_pred):.4f}")

# ===================== (b) Compare with scikit-learn ===========================
sk_dt = DecisionTreeRegressor(random_state=42)   # default settings
sk_dt.fit(Xtr, ytr)
y_pred_sk = pd.Series(sk_dt.predict(Xte), index=yte.index)

print("\n=== (b) scikit-learn DecisionTreeRegressor ===")
print(f"MSE : {mse(yte, y_pred_sk):.4f}")
print(f"RMSE: {rmse(yte, y_pred_sk):.4f}")
print(f"MAE : {mae(yte, y_pred_sk):.4f}")
print(f"R^2 : {r2(yte, y_pred_sk):.4f}")

# -------------------- Plot: Predicted vs True MPG --------------------
lims = [min(yte.min(), y_pred.min(), y_pred_sk.min()) - 1,
        max(yte.max(), y_pred.max(), y_pred_sk.max()) + 1]

plt.figure(figsize=(11,4))
# Our tree
plt.subplot(1,2,1)
plt.scatter(yte, y_pred, s=15)
plt.plot(lims, lims, linestyle="--")
plt.xlabel("True MPG"); plt.ylabel("Predicted MPG")
plt.title("Our DecisionTree")
plt.xlim(lims); plt.ylim(lims)

# sklearn tree
plt.subplot(1,2,2)
plt.scatter(yte, y_pred_sk, s=15)
plt.plot(lims, lims, linestyle="--")
plt.xlabel("True MPG"); plt.ylabel("Predicted MPG")
plt.title("sklearn DecisionTreeRegressor")
plt.xlim(lims); plt.ylim(lims)

plt.tight_layout()
plt.show()
