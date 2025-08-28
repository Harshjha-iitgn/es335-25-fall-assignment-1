"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these functions are here to simply help you.
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd


# ---------------------------
# Encoding / Type utilities
# ---------------------------

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Perform one-hot encoding on non-numeric (object/category) columns.
    Numeric columns are kept as-is but cast to float.
    Returns a new DataFrame with all features as real-valued (float).

    Note:
    - We do NOT drop the first level (drop_first=False) to keep splits simple.
    - If you have integer-coded categoricals, convert to pandas 'category'
      before calling this function; otherwise they will be treated as numeric.
    """
    X = X.copy()
    non_num_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if non_num_cols:
        X = pd.get_dummies(X, columns=non_num_cols, drop_first=False, dtype=float)
    # ensure all numeric types are float
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype(float)
    return X


def check_ifreal(y: pd.Series) -> bool:
    """
    Check if the target looks real-valued (regression) or discrete (classification).

    Heuristic:
    - If dtype is float/complex and number of unique values is large
      ( > max(10, sqrt(n)) ), treat as real-valued.
    - Otherwise treat as discrete (classification).
    """
    s = pd.Series(y).dropna()
    n = len(s)
    nunq = s.nunique()
    if s.dtype.kind in "fc" and nunq > max(10, int(np.sqrt(max(n, 1)))):
        return True
    return False


# ---------------------------
# Impurity measures
# ---------------------------

def entropy(Y: pd.Series) -> float:
    """
    Shannon entropy (base-2) for discrete labels.
    """
    s = pd.Series(Y).dropna()
    if len(s) == 0:
        return 0.0
    p = s.value_counts(normalize=True).values.astype(float)
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log2(p)))


def gini_index(Y: pd.Series) -> float:
    """
    Gini index for discrete labels.
    """
    s = pd.Series(Y).dropna()
    if len(s) == 0:
        return 0.0
    p = s.value_counts(normalize=True).values.astype(float)
    return float(1.0 - np.sum(p ** 2))


def _node_mse(Y: pd.Series) -> float:
    """
    Per-node MSE proxy for regression impurity (variance).
    """
    s = pd.Series(Y).dropna().astype(float)
    if len(s) == 0:
        return 0.0
    return float(np.var(s))


def _impurity(Y: pd.Series, criterion: str) -> float:
    """
    Criterion can be:
      - 'entropy'  (information gain for classification)
      - 'gini'     (gini reduction for classification)
      - 'mse'      (variance/MSE reduction for regression)
    Also accept aliases: 'information_gain' -> 'entropy', 'gini_index' -> 'gini'
    """
    crit = criterion.lower()
    if crit == "information_gain":
        crit = "entropy"
    if crit == "gini_index":
        crit = "gini"

    if crit == "entropy":
        return entropy(Y)
    elif crit == "gini":
        return gini_index(Y)
    elif crit == "mse":
        return _node_mse(Y)
    else:
        raise ValueError("criterion must be one of: 'entropy'|'gini'|'mse' (or aliases 'information_gain'|'gini_index')")


# ---------------------------
# Information gain helpers
# ---------------------------

def _weighted_child_impurity(parent: pd.Series, splits: List[pd.Series], criterion: str) -> float:
    """
    Compute weighted child impurity given a list of child target series.
    """
    n = len(parent)
    total = 0.0
    for child in splits:
        w = len(child) / max(n, 1)
        total += w * _impurity(child, criterion)
    return total


def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Compute impurity reduction for a DISCRETE attribute split (multiway):
    IG = impurity(parent) - sum_k (|Y_k|/|Y|) * impurity(Y_k)

    For REAL-valued attributes (continuous), prefer using `opt_split_attribute`
    which scans thresholds and returns the best gain and threshold.

    criterion: 'entropy' | 'gini' | 'mse' (or aliases 'information_gain'|'gini_index')
    """
    parent_imp = _impurity(Y, criterion)
    # group by attribute values (discrete)
    splits = [Y[attr == v] for v in pd.Series(attr).dropna().unique()]
    if len(splits) == 0:
        return 0.0
    children = _weighted_child_impurity(Y, splits, criterion)
    return float(parent_imp - children)


# ---------------------------
# Split search
# ---------------------------

def _best_threshold_for_numeric(x: pd.Series, y: pd.Series, criterion: str, min_leaf: int = 1) -> Tuple[Optional[float], float]:
    """
    Find best threshold for a numeric feature by scanning midpoints between sorted unique values.
    Returns (best_threshold, best_gain). If no valid split: (None, -inf).
    """
    x = pd.Series(x).astype(float)
    y = pd.Series(y)
    order = np.argsort(x.values)
    x_sorted = x.values[order]
    y_sorted = y.values[order]

    uniq = np.unique(x_sorted)
    if uniq.size <= 1:
        return None, float("-inf")

    # candidate thresholds: midpoints between consecutive distinct values
    thresholds = (uniq[:-1] + uniq[1:]) / 2.0
    best_thr, best_gain = None, float("-inf")

    parent_imp = _impurity(y_sorted, criterion)
    n = len(y_sorted)

    for thr in thresholds:
        left_mask = x_sorted <= thr
        right_mask = ~left_mask
        nL, nR = left_mask.sum(), right_mask.sum()
        if nL < min_leaf or nR < min_leaf:
            continue
        yL = y_sorted[left_mask]
        yR = y_sorted[right_mask]
        # gain = parent - weighted children
        w_child = (nL / n) * _impurity(yL, criterion) + (nR / n) * _impurity(yR, criterion)
        gain = parent_imp - w_child
        if gain > best_gain:
            best_gain, best_thr = float(gain), float(thr)

    return best_thr, best_gain


def opt_split_attribute(
    X: pd.DataFrame,
    y: pd.Series,
    criterion: str,
    features: pd.Series,
    min_leaf: int = 1
) -> Tuple[Optional[str], Optional[float], bool, float]:
    """
    Find the optimal attribute to split upon.

    Parameters
    ----------
    X : DataFrame
        Feature matrix.
    y : Series
        Target vector.
    criterion : str
        'entropy' | 'gini' | 'mse' (aliases 'information_gain'|'gini_index' accepted)
    features : pd.Series | list-like
        Column names to consider for splitting.
    min_leaf : int
        Minimum number of samples in each child after split.

    Returns
    -------
    best_attr : Optional[str]
        The column name of the best attribute, or None if no split.
    best_value : Optional[float]
        If numeric feature → the best threshold (float).
        If discrete feature → the category value chosen for binary split (equals-vs-others).
    is_numeric : bool
        True if the split is on a numeric column.
    best_gain : float
        The impurity reduction obtained by the chosen split.

    Notes
    -----
    - For numeric columns: binary split at threshold (x <= thr) vs (x > thr).
    - For discrete columns: binary split at value (x == v) vs (x != v),
      scanning each category v and picking the best.
    - If your pipeline one-hot encodes discrete features before fit(),
      all features will be numeric and this function will run the numeric path.
    """
    if not isinstance(features, (list, tuple, pd.Series, pd.Index)):
        features = [features]
    features = list(features)

    best_attr, best_value, best_gain = None, None, float("-inf")
    best_is_numeric = True

    for col in features:
        col_series = X[col]
        if pd.api.types.is_numeric_dtype(col_series):
            # numeric path: scan thresholds
            thr, gain = _best_threshold_for_numeric(col_series, y, criterion, min_leaf=min_leaf)
            if gain > best_gain:
                best_attr, best_value, best_gain = col, thr, gain
                best_is_numeric = True
        else:
            # discrete path: try binary split for each category value
            parent_imp = _impurity(y, criterion)
            vals = pd.Series(col_series).astype("category").cat.categories.tolist()
            for v in vals:
                left = y[col_series == v]
                right = y[col_series != v]
                if len(left) < min_leaf or len(right) < min_leaf:
                    continue
                n = len(y)
                w_child = (len(left) / n) * _impurity(left, criterion) + (len(right) / n) * _impurity(right, criterion)
                gain = parent_imp - w_child
                if gain > best_gain:
                    best_attr, best_value, best_gain = col, v, float(gain)
                    best_is_numeric = False

    return best_attr, best_value, best_is_numeric, float(best_gain)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    attribute: str,
    value
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split the data according to an attribute and a value.

    If attribute is numeric:
        left  = X[attribute] <= value
        right = X[attribute] >  value

    If attribute is discrete (object/category):
        left  = X[attribute] == value
        right = X[attribute] != value

    Returns (X_left, y_left, X_right, y_right)
    """
    col = X[attribute]
    if pd.api.types.is_numeric_dtype(col):
        left_mask = col <= float(value)
        right_mask = ~left_mask
    else:
        left_mask = col == value
        right_mask = ~left_mask

    X_left, y_left = X[left_mask], y[left_mask]
    X_right, y_right = X[right_mask], y[right_mask]
    return X_left, y_left, X_right, y_right
