import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

DB = Path("data/estimates.db")

SCHEMA = """CREATE TABLE IF NOT EXISTS estimates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TEXT, description TEXT, domain TEXT, size REAL, unit TEXT,
    team INTEGER, budget REAL, effort_hours REAL, duration_months REAL,
    best_guess REAL, p10 REAL, p50 REAL, p90 REAL, actual_cost REAL)"""


def _connect():
    DB.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute(SCHEMA)
    return con


def save_estimate(description, domain, size, unit, team, budget, est, sim):
    con = _connect()
    try:
        cur = con.execute(
            "INSERT INTO estimates (created, description, domain, size, unit, team, "
            "budget, effort_hours, duration_months, best_guess, p10, p50, p90) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), description, domain, size,
             unit, team, budget, est["effort_hours"], est["duration_months"],
             est["total_cost"], sim["P10"], sim["P50"], sim["P90"]))
        con.commit()
        return cur.lastrowid
    finally:
        con.close()


def list_estimates():
    con = _connect()
    try:
        return pd.read_sql_query(
            "SELECT id, created, domain, size, unit, team, budget, best_guess, "
            "p10, p50, p90, actual_cost FROM estimates ORDER BY id DESC", con)
    finally:
        con.close()
def set_actual(estimate_id, actual_cost):
    con = _connect()
    try:
        con.execute("UPDATE estimates SET actual_cost=? WHERE id=?",
                    (actual_cost, estimate_id))
        con.commit()
    finally:
        con.close()


def kpis():
    df = list_estimates()
    df["actual_cost"] = pd.to_numeric(df["actual_cost"], errors="coerce")
    d = df.dropna(subset=["actual_cost"])
    d = d[d["actual_cost"] > 0]
    if d.empty:
        return None
    err = (d["best_guess"] - d["actual_cost"]) / d["actual_cost"]
    inside = (d["actual_cost"] >= d["p10"]) & (d["actual_cost"] <= d["p90"])
    return {
        "projects with actual cost": len(d),
        "MMRE (cost error)": round(float(err.abs().mean()), 2),
        "PRED(25) %": round(float((err.abs() <= 0.25).mean() * 100), 1),
        "delivered within budget %": round(float((d["actual_cost"] <= d["budget"]).mean() * 100), 1),
        "actual inside P10-P90 %": round(float(inside.mean() * 100), 1),
        "bias % (+ = overestimate)": round(float(err.mean() * 100), 1),
    }