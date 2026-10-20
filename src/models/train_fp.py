import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from src.data.load import load_file
from src.metrics import mae, mmre, mdmre, pred, bias

def cv_pred(model, X, ylog):
    p = cross_val_predict(model, X, ylog, cv=KFold(10, shuffle=True, random_state=0))
    return np.exp(p)

def row(exp, name, y, p):
    return {"experiment": exp, "model": name, "MAE_h": mae(y, p), "MMRE": mmre(y, p),
            "MdMRE": mdmre(y, p), "PRED25": pred(y, p), "bias_h": bias(y, p)}

rows = []

# A: all function-point datasets, size only
df = pd.read_csv("data/processed/unified.csv")
fp = df[df["size_unit"] == "fp"].reset_index(drop=True)
assert "china" in set(fp["dataset"]), "china missing: re-run python -m src.data.clean"
print(fp.groupby("dataset").size(), "\n")

X = np.log(fp[["size_value"]].to_numpy(float))
y = fp["effort_hours"].to_numpy(float)
pA = {}
for name, m in [("loglog_size", LinearRegression()),
                ("gradient_boosting", GradientBoostingRegressor(
                    n_estimators=200, max_depth=2, learning_rate=0.05, random_state=0))]:
    pA[name] = cv_pred(m, X, np.log(y))
    rows.append(row("A_all_fp_size_only", name, y, pA[name]))

fp["ok"] = np.abs(pA["loglog_size"] - y) / y <= 0.25
print("loglog PRED25 % per dataset:\n", (fp.groupby("dataset")["ok"].mean() * 100).round(1), "\n")

# B: China only, size vs rich features (leaky columns left out)
c = load_file("data/raw/china/china.arff")
c = c[c["Effort"] > 0].reset_index(drop=True)
feats = ["AFP", "Input", "Output", "Enquiry", "File", "Interface",
         "Added", "Changed", "Deleted", "Resource", "Dev.Type"]
yc = c["Effort"].to_numpy(float)
Xs = np.log(c[["AFP"]].to_numpy(float))
Xr = np.log1p(c[feats].clip(lower=0).to_numpy(float))
rows.append(row("B_china", "loglog_size", yc, cv_pred(LinearRegression(), Xs, np.log(yc))))
rows.append(row("B_china", "random_forest_rich", yc,
                cv_pred(RandomForestRegressor(n_estimators=300, random_state=0), Xr, np.log(yc))))
rows.append(row("B_china", "gradient_boosting_rich", yc,
                cv_pred(GradientBoostingRegressor(n_estimators=200, max_depth=3,
                        learning_rate=0.05, random_state=0), Xr, np.log(yc))))

res = pd.DataFrame(rows).round(2)
print(res.to_string(index=False))
res.to_csv("reports/fp_models.csv", index=False)