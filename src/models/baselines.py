import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from src.metrics import mae, mmre, mdmre, pred, bias

df = pd.read_csv("data/processed/unified.csv")

def loo_predict(X, y, model_fn):
    # leave-one-out: each project is predicted by a model that never saw it
    out = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        out[te] = model_fn(X[tr], y[tr], X[te])
    return out

def median_model(Xtr, ytr, Xte):
    return np.full(len(Xte), np.median(ytr))

def loglog_model(Xtr, ytr, Xte):
    m = LinearRegression().fit(np.log(Xtr), np.log(ytr))
    return np.exp(m.predict(np.log(Xte)))

rows = []
for unit, g in df.groupby("size_unit"):
    X = g[["size_value"]].to_numpy(float)
    y = g["effort_hours"].to_numpy(float)
    for name, fn in [("median", median_model), ("loglog_size", loglog_model)]:
        p = loo_predict(X, y, fn)
        rows.append({"family": unit, "model": name,
                     "MAE": mae(y, p), "MMRE": mmre(y, p),
                     "MdMRE": mdmre(y, p), "PRED25": pred(y, p),
                     "bias": bias(y, p)})

res = pd.DataFrame(rows).round(2)
print(res.to_string(index=False))
res.to_csv("reports/baseline_results.csv", index=False)