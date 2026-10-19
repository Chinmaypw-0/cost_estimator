import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from src.data.schema import MULTIPLIERS
from src.metrics import mae, mmre, mdmre, pred, bias

nasa = pd.read_csv("data/processed/nasa93_cocomo.csv")
coc = pd.read_csv("data/processed/cocomo811_cocomo.csv")
df = pd.concat([nasa, coc], ignore_index=True)

eaf = df[list(MULTIPLIERS)].prod(axis=1).to_numpy()
kloc = df["kloc"].to_numpy(float)
y = df["effort_pm"].to_numpy(float)

def loo(use_eaf):
    adj = y / eaf if use_eaf else y
    out = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(kloc):
        m = LinearRegression().fit(np.log(kloc[tr]).reshape(-1, 1), np.log(adj[tr]))
        p = np.exp(m.predict(np.log(kloc[te]).reshape(-1, 1)))
        out[te] = p * (eaf[te] if use_eaf else 1)
    return out

rows = []
for name, flag in [("size_only", False), ("cocomo_with_multipliers", True)]:
    p = loo(flag)
    rows.append({"model": name, "MAE_pm": mae(y, p), "MMRE": mmre(y, p),
                 "MdMRE": mdmre(y, p), "PRED25": pred(y, p), "bias_pm": bias(y, p)})

res = pd.DataFrame(rows).round(2)
print(res.to_string(index=False))
res.to_csv("reports/cocomo_results.csv", index=False)