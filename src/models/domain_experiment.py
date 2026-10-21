import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from src.data.schema import MULTIPLIERS
from src.metrics import mmre, mdmre, pred

nasa = pd.read_csv("data/processed/nasa93_cocomo.csv").assign(src="nasa93")
coc = pd.read_csv("data/processed/cocomo811_cocomo.csv").assign(src="cocomo811")
df = pd.concat([nasa, coc], ignore_index=True)

eaf = df[list(MULTIPLIERS)].prod(axis=1).to_numpy()
size = np.log(df["kloc"].to_numpy(float))
y = df["effort_pm"].to_numpy(float)
adj = np.log(y / eaf)
is_nasa = (df["src"] == "nasa93").to_numpy(float)

def loo(X, rows):
    # each row is predicted by a model trained on the other rows in `rows`
    idx = np.array(rows)
    out = {}
    for i in idx:
        tr = idx[idx != i]
        m = LinearRegression().fit(X[tr], adj[tr])
        out[i] = np.exp(m.predict(X[[i]])[0]) * eaf[i]
    return out

X1 = size.reshape(-1, 1)
X2 = np.column_stack([size, is_nasa])
all_rows = list(range(len(df)))

preds = {"general": loo(X1, all_rows),
         "general+domain_feature": loo(X2, all_rows)}
spec = {}
for s in ["nasa93", "cocomo811"]:
    spec.update(loo(X1, list(df.index[df["src"] == s])))
preds["specialist"] = spec

rows = []
for name, p in preds.items():
    for s in ["nasa93", "cocomo811"]:
        idx = df.index[df["src"] == s]
        a = y[idx]
        b = np.array([p[i] for i in idx])
        rows.append({"model": name, "group": s, "n": len(idx),
                     "MMRE": mmre(a, b), "MdMRE": mdmre(a, b), "PRED25": pred(a, b)})

res = pd.DataFrame(rows).round(2)
print(res.to_string(index=False))
res.to_csv("reports/domain_experiment.csv", index=False)