"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output

Note: Model accepts real inputs only. If you have discrete/categorical inputs,
convert them to one-hot (0/1 float) BEFORE calling fit().
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # (not required for ASCII plot, kept to match skeleton)

np.random.seed(42)


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # used for classification
    max_depth: int                                        # maximum tree depth

    def __init__(self, criterion="information_gain", max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = 2
        self.min_leaf = 1
        self.root_ = None
        self.task_ = None  # 'classification' or 'regression'

    # ---------------------- core helpers ----------------------

    def _detect_task(self, y: np.ndarray) -> str:
        # Heuristic: many distinct float targets -> regression; else classification
        nunq = np.unique(y).size
        if (y.dtype.kind == "f" and nunq > 10) or (y.dtype.kind in "fc" and nunq > max(10, int(np.sqrt(len(y))))):
            return "regression"
        return "classification"

    def _leaf_value(self, y: np.ndarray):
        if self.task_ == "classification":
            vals, cnts = np.unique(y, return_counts=True)
            return vals[np.argmax(cnts)]
        else:
            return float(np.mean(y))

    # impurities
    def _entropy(self, y):
        vals, cnts = np.unique(y, return_counts=True)
        p = cnts / cnts.sum()
        return float(-np.sum(p * np.log2(p + 1e-12)))

    def _gini(self, y):
        vals, cnts = np.unique(y, return_counts=True)
        p = cnts / cnts.sum()
        return float(1.0 - np.sum(p ** 2))

    def _mse_node(self, y):
        # variance is proportional to MSE inside a node
        return float(np.var(y))

    def _impurity(self, y):
        if self.task_ == "classification":
            if self.criterion == "information_gain":
                return self._entropy(y)
            elif self.criterion == "gini_index":
                return self._gini(y)
            else:
                raise ValueError("criterion must be 'information_gain' or 'gini_index' for classification")
        else:
            return self._mse_node(y)

    def _gain(self, y_parent, y_left, y_right):
        n = len(y_parent)
        if len(y_left) == 0 or len(y_right) == 0:
            return -np.inf
        wL, wR = len(y_left) / n, len(y_right) / n
        return self._impurity(y_parent) - (wL * self._impurity(y_left) + wR * self._impurity(y_right))

    def _best_split_feature(self, x_col: np.ndarray, y: np.ndarray):
        # Real-valued feature split by thresholds at midpoints between unique sorted values
        order = np.argsort(x_col)
        x_sorted, y_sorted = x_col[order], y[order]
        uniq = np.unique(x_sorted)
        if uniq.size <= 1:
            return None, -np.inf, None, None

        thresholds = (uniq[:-1] + uniq[1:]) / 2.0
        best_thr, best_gain = None, -np.inf
        best_left_idx, best_right_idx = None, None

        for thr in thresholds:
            left_mask = x_sorted <= thr
            right_mask = ~left_mask
            if left_mask.sum() < self.min_leaf or right_mask.sum() < self.min_leaf:
                continue
            g = self._gain(y_sorted, y_sorted[left_mask], y_sorted[right_mask])
            if g > best_gain:
                best_gain = g
                best_thr = thr
                # store masks w.r.t original order
                left_idx_sorted = left_mask
                right_idx_sorted = right_mask
                # convert to original index space
                left_orig = order[left_idx_sorted]
                right_orig = order[right_idx_sorted]
                best_left_idx, best_right_idx = left_orig, right_orig

        return best_thr, best_gain, best_left_idx, best_right_idx

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        best = {"feat": None, "thr": None, "gain": -np.inf, "L": None, "R": None}

        for j in range(n_features):
            thr, gain, L_idx, R_idx = self._best_split_feature(X[:, j], y)
            if gain > best["gain"]:
                best.update({"feat": j, "thr": thr, "gain": gain, "L": L_idx, "R": R_idx})

        return best

    # ---------------------- tree construction ----------------------

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int):
        # stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           len(y) < self.min_samples_split or \
           np.unique(y).size == 1:
            return {"is_leaf": True, "value": self._leaf_value(y)}

        best = self._best_split(X, y)
        if best["feat"] is None or best["gain"] <= 0 or best["L"] is None or best["R"] is None:
            return {"is_leaf": True, "value": self._leaf_value(y)}

        node = {
            "is_leaf": False,
            "feature": int(best["feat"]),
            "threshold": float(best["thr"]),
            "left": self._build(X[best["L"]], y[best["L"]], depth + 1),
            "right": self._build(X[best["R"]], y[best["R"]], depth + 1)
        }
        return node

    # ---------------------- public API ----------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train and construct the decision tree.
        X: pd.DataFrame of real-valued features (one-hot if originally discrete)
        y: pd.Series of targets (discrete labels OR real values)
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)

        self.task_ = self._detect_task(y_arr)
        self.root_ = self._build(X_arr, y_arr, depth=0)
        return None

    def _predict_row(self, row: np.ndarray, node):
        if node["is_leaf"]:
            return node["value"]
        # split rule: left if <= threshold, right otherwise
        if row[node["feature"]] <= node["threshold"]:
            return self._predict_row(row, node["left"])
        else:
            return self._predict_row(row, node["right"])

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Run the decision tree on test inputs.
        Returns a pd.Series with the same index as X.
        """
        if self.root_ is None:
            raise RuntimeError("Call fit() before predict().")
        X_arr = np.asarray(X, dtype=float)
        preds = [self._predict_row(r, self.root_) for r in X_arr]
        return pd.Series(preds, index=X.index)

    # ---------------------- ASCII plot ----------------------

    def _format_node(self, node, depth=0):
        pad = "    " * depth
        if node["is_leaf"]:
            return f"{pad}{node['value']}\n"

        # Show condition as ?(Xk > thr), and print Y-branch (right) first like the example
        s = f"{pad}?(X{node['feature']+1} > {node['threshold']:.5g})\n"
        s += f"{pad}    Y: " + self._format_node(node["right"], depth + 2)
        s += f"{pad}    N: " + self._format_node(node["left"], depth + 2)
        return s

    def plot(self) -> None:
        """
        Print an ASCII visualization of the tree, e.g.:

        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        """
        if self.root_ is None:
            print("(empty tree: call fit() first)")
            return
        print(self._format_node(self.root_), end="")
