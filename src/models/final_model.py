import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from src.data.schema import MULTIPLIERS

Path("models").mkdir(exist_ok=True)

# KLOC family: size + 15 COCOMO multipliers (target: person-months)
nasa = pd.read_csv("data/processed/nasa93_cocomo.csv")
coc = pd.read_csv("data/processed/cocomo811_cocomo.csv")
k = pd.concat([nasa, coc], ignore_index=True)
eaf = k[list(MULTIPLIERS)].prod(axis=1).to_numpy()
Xk = np.log(k[["kloc"]].to_numpy(float))
yk = np.log(k["effort_pm"].to_numpy(float) / eaf)

resid_k = yk - cross_val_predict(LinearRegression(), Xk, yk, cv=LeaveOneOut())
model_k = LinearRegression().fit(Xk, yk)

# Function-point family: size only (target: person-hours)
df = pd.read_csv("data/processed/unified.csv")
fp = df[df["size_unit"] == "fp"]
Xf = np.log(fp[["size_value"]].to_numpy(float))
yf = np.log(fp["effort_hours"].to_numpy(float))

cv = KFold(10, shuffle=True, random_state=0)
resid_f = yf - cross_val_predict(LinearRegression(), Xf, yf, cv=cv)
model_f = LinearRegression().fit(Xf, yf)

joblib.dump(model_k, "models/kloc_model.joblib")
joblib.dump(model_f, "models/fp_model.joblib")
json.dump({"kloc": resid_k.tolist(), "fp": resid_f.tolist()},
          open("models/residuals.json", "w"))

for name, m, r in [("kloc", model_k, resid_k), ("fp", model_f, resid_f)]:
    lo, hi = np.exp(np.percentile(r, [10, 90]))
    print(f"{name}: effort = exp({m.intercept_:.2f}) * size^{m.coef_[0]:.2f}"
          f" | 80% of actuals fall between {lo:.2f}x and {hi:.2f}x of the estimate")