from typing import Union, Dict, Tuple, Iterable
import numpy as np
import pandas as pd

# ----------------------------
# Classification metrics
# ----------------------------

def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean())

def precision_recall_per_class(
    y_true,
    y_pred,
    labels: Iterable = None,
    eps: float = 1e-12
) -> Tuple[Dict[Union[int, float, str], float], Dict[Union[int, float, str], float], float, float]:
    """
    Returns:
      prec_dict: {class -> precision}
      rec_dict : {class -> recall}
      prec_macro: mean precision over classes
      rec_macro : mean recall over classes
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if labels is None:
        labels = np.unique(y_true)

    prec, rec = {}, {}
    for c in labels:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec[c] = tp / (tp + fp + eps)
        rec[c]  = tp / (tp + fn + eps)

    prec_macro = float(np.mean(list(prec.values()))) if len(prec) else 0.0
    rec_macro  = float(np.mean(list(rec.values()))) if len(rec) else 0.0
    return prec, rec, prec_macro, rec_macro

def precision_for_class(y_true, y_pred, cls, eps: float = 1e-12) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_true == cls) & (y_pred == cls))
    fp = np.sum((y_true != cls) & (y_pred == cls))
    return float(tp / (tp + fp + eps))

def recall_for_class(y_true, y_pred, cls, eps: float = 1e-12) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_true == cls) & (y_pred == cls))
    fn = np.sum((y_true == cls) & (y_pred != cls))
    return float(tp / (tp + fn + eps))

# ----------------------------
# Regression metrics
# ----------------------------

def mse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))

def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def r2(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))
