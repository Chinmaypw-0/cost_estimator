import numpy as np

def mre(actual, pred_):
    actual, pred_ = np.asarray(actual, float), np.asarray(pred_, float)
    return np.abs(actual - pred_) / np.abs(actual)

def mmre(actual, pred_):
    return float(np.mean(mre(actual, pred_)))

def mdmre(actual, pred_):
    return float(np.median(mre(actual, pred_)))

def pred(actual, pred_, level=0.25):
    return float(np.mean(mre(actual, pred_) <= level) * 100)

def mae(actual, pred_):
    return float(np.mean(np.abs(np.asarray(actual, float) - np.asarray(pred_, float))))

def bias(actual, pred_):
    """Mean signed error: positive = overestimating, negative = underestimating."""
    return float(np.mean(np.asarray(pred_, float) - np.asarray(actual, float)))