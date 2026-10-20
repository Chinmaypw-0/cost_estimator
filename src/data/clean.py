from pathlib import Path
import numpy as np
import pandas as pd
from src.data.load import RAW, load_file
from src.data.schema import (HOURS_PER_PM, ALBRECHT_HOURS_PER_UNIT,
                             UNIFIED_COLS, DOMAIN_GROUP, MULTIPLIERS)

PROC = Path("data/processed")

def _load(name):
    return load_file(RAW / name / f"{name}.arff")

def _frame(name, size, unit, effort_hours, duration=np.nan, team_exp=np.nan):
    return pd.DataFrame({
        "dataset": name,
        "domain_group": DOMAIN_GROUP[name],
        "size_value": size,
        "size_unit": unit,
        "duration_months": duration,
        "team_experience": team_exp,
        "effort_hours": effort_hours,
    })

def albrecht():
    df = _load("albrecht")
    return _frame("albrecht", df["AdjFP"], "fp", df["Effort"] * ALBRECHT_HOURS_PER_UNIT)

def cocomo811():
    df = _load("cocomo811")
    return _frame("cocomo811", df["loc"], "kloc", df["actual"] * HOURS_PER_PM)

def desharnais():
    df = _load("desharnais").replace(-1, np.nan)      # -1 means missing in this dataset
    df["Language"] = pd.to_numeric(df["Language"], errors="coerce")
    before = len(df)
    df = df.dropna()                                   # standard 77 complete cases
    print(f"desharnais: {before} -> {len(df)} rows after dropping incomplete")
    return _frame("desharnais", df["PointsAjust"], "fp", df["Effort"],
                  duration=df["Length"], team_exp=df["TeamExp"])

def kemerer():
    df = _load("kemerer")
    return _frame("kemerer", df["AdjFP"], "fp", df["EffortMM"] * HOURS_PER_PM,
                  duration=df["Duration"])

def maxwell():
    df = _load("maxwell")
    return _frame("maxwell", df["Size"], "fp", df["Effort"], duration=df["Duration"])

def nasa93():
    df = _load("nasa93")
    return _frame("nasa93", df["equivphyskloc"], "kloc", df["act_effort"] * HOURS_PER_PM)

def china():
    df = _load("china")
    return _frame("china", df["AFP"], "fp", df["Effort"], duration=df["Duration"])

def build_unified():
    out = pd.concat([f() for f in (albrecht, cocomo811, desharnais,china,
                                   kemerer, maxwell, nasa93)], ignore_index=True)
    out = out[UNIFIED_COLS]
    bad = out["size_value"].isna() | out["effort_hours"].isna() \
        | (out["size_value"] <= 0) | (out["effort_hours"] <= 0)
    print(f"dropping {int(bad.sum())} rows with missing/non-positive size or effort")
    return out[~bad].reset_index(drop=True)

def cocomo_tables():
    """Numeric multiplier tables (for the COCOMO II baseline in Step 4)."""
    n = _load("nasa93")
    cols = {}
    for m, mapping in MULTIPLIERS.items():
        cols[m] = n[m].astype(str).str.strip().str.lower().map(mapping)
    nasa = pd.DataFrame(cols)
    nasa["kloc"] = n["equivphyskloc"]
    nasa["effort_pm"] = n["act_effort"]
    nasa["mode"] = n["mode"]
    unmapped = nasa[list(MULTIPLIERS)].isna().sum().sum()
    print(f"nasa93: unmapped multiplier cells = {int(unmapped)} (should be 0)")

    c = _load("cocomo811")
    coc = c[list(MULTIPLIERS)].copy()
    coc["kloc"] = c["loc"]
    coc["effort_pm"] = c["actual"]
    coc["mode"] = np.nan
    return nasa, coc

if __name__ == "__main__":
    PROC.mkdir(parents=True, exist_ok=True)
    unified = build_unified()
    unified.to_csv(PROC / "unified.csv", index=False)
    nasa, coc = cocomo_tables()
    nasa.to_csv(PROC / "nasa93_cocomo.csv", index=False)
    coc.to_csv(PROC / "cocomo811_cocomo.csv", index=False)
    print("\nRows per dataset:\n", unified.groupby("dataset").size())
    print("\nEffort (hours) summary:\n",
          unified.groupby("dataset")["effort_hours"].describe()[["min", "50%", "max"]])
    print("\nRows per size family:\n", unified.groupby("size_unit").size())