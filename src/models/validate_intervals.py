import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from src.data.schema import MULTIPLIERS

def check(X, y):
    hit, under = [], []
    for tr, te in KFold(10, shuffle=True, random_state=0).split(X):
        inner = y[tr] - cross_val_predict(LinearRegression(), X[tr], y[tr], cv=5)
        lo, hi = np.percentile(inner, [10, 90])
        r = y[te] - LinearRegression().fit(X[tr], y[tr]).predict(X[te])
        hit += list((r >= lo) & (r <= hi))
        under += list(r > 0)           # actual effort above the estimate
    return round(np.mean(hit) * 100, 1), round(np.mean(under) * 100, 1)

# function-point family
df = pd.read_csv("data/processed/unified.csv")
fp = df[df["size_unit"] == "fp"]
cov_fp, und_fp = check(np.log(fp[["size_value"]].to_numpy(float)),
                       np.log(fp["effort_hours"].to_numpy(float)))

# KLOC family (size + COCOMO factors)
k = pd.concat([pd.read_csv("data/processed/nasa93_cocomo.csv"),
               pd.read_csv("data/processed/cocomo811_cocomo.csv")], ignore_index=True)
eaf = k[list(MULTIPLIERS)].prod(axis=1).to_numpy()
cov_k, und_k = check(np.log(k[["kloc"]].to_numpy(float)),
                     np.log(k["effort_pm"].to_numpy(float) / eaf))

print(pd.DataFrame({
    "family": ["fp", "kloc"],
    "range_coverage_%": [cov_fp, cov_k],
    "actual_above_estimate_%": [und_fp, und_k],
}).to_string(index=False))
print("\nTarget: coverage near 80, actual_above_estimate near 50")