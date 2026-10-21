import joblib
import numpy as np
import yaml
from src.data.schema import MULTIPLIERS, HOURS_PER_PM

ROLE_SPLIT = {"developer": 0.55, "tester": 0.20, "project_manager": 0.10,
              "designer": 0.10, "devops": 0.05}

BASE_TEAM = 3                # team size with no extra overhead
OVERHEAD_PER_EXTRA = 0.05    # +5% effort per person beyond the base team

def team_overhead(team_size):
    return 1 + OVERHEAD_PER_EXTRA * max(0, team_size - BASE_TEAM)

def load_rates(path="config/rates.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def estimate_effort_hours(size, unit, multipliers=None):
    if unit == "kloc":
        model = joblib.load("models/kloc_model.joblib")
        eaf = 1.0
        for m in MULTIPLIERS:
            eaf *= (multipliers or {}).get(m, 1.0)   # missing factor = nominal (1.0)
        pm = float(np.exp(model.predict(np.log([[size]]))[0])) * eaf
        return pm * HOURS_PER_PM
    model = joblib.load("models/fp_model.joblib")
    return float(np.exp(model.predict(np.log([[size]]))[0]))

def cost_estimate(size, unit, team_size=5, multipliers=None, rates=None):
    rates = rates or load_rates()
    hours = estimate_effort_hours(size, unit, multipliers) * team_overhead(team_size)
    months = hours / (team_size * HOURS_PER_PM)
    roles = {}
    for role, share in ROLE_SPLIT.items():
        h = hours * share
        roles[role] = {"hours": round(h), "cost": round(h * rates["roles"][role])}
    personnel = sum(r["cost"] for r in roles.values())
    cloud = rates["cloud_per_month"] * months
    tooling = rates["tooling_per_project"]
    other = rates["other_per_project"]
    return {
        "effort_hours": round(hours),
        "duration_months": round(months, 1),
        "roles": roles,
        "personnel_cost": personnel,
        "cloud_cost": round(cloud),
        "tooling_cost": tooling,
        "other_cost": other,
        "total_cost": round(personnel + cloud + tooling + other),
    }

if __name__ == "__main__":
    import json
    print("50 KLOC project, team of 5:")
    print(json.dumps(cost_estimate(50, "kloc", team_size=5), indent=2))
    print("\n300 function-point project, team of 4:")
    print(json.dumps(cost_estimate(300, "fp", team_size=4), indent=2))