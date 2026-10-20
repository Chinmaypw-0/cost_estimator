import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.data.schema import MULTIPLIERS
from src.metrics import mae, mmre, mdmre, pred, bias

nasa = pd.read_csv("data/processed/nasa93_cocomo.csv")
coc = pd.read_csv("data/processed/cocomo811_cocomo.csv")
df = pd.concat([nasa, coc], ignore_index=True)

# features: log size + log of each COCOMO multiplier; target: log effort (person-months)
X = np.column_stack([np.log(df["kloc"])] + [np.log(df[m]) for m in MULTIPLIERS])
y = df["effort_pm"].to_numpy(float)
ylog = np.log(y)

models = {
    "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "random_forest": RandomForestRegressor(n_estimators=300, random_state=0),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=200, max_depth=2, learning_rate=0.05, random_state=0),
}

rows = []
for name, model in models.items():
    p = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        model.fit(X[tr], ylog[tr])
        p[te] = np.exp(model.predict(X[te]))
    rows.append({"model": name, "MAE_pm": mae(y, p), "MMRE": mmre(y, p),
                 "MdMRE": mdmre(y, p), "PRED25": pred(y, p), "bias_pm": bias(y, p)})

res = pd.DataFrame(rows).round(2)
print(res.to_string(index=False))
res.to_csv("reports/kloc_models.csv", index=False)