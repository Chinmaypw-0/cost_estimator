import json
import numpy as np
from src.costing.cost_engine import ROLE_SPLIT, estimate_effort_hours, load_rates, team_overhead
from src.data.schema import HOURS_PER_PM

def simulate(size, unit, team_size=5, multipliers=None, n=10000, budget=None, seed=0):
    rng = np.random.default_rng(seed)
    rates = load_rates()
    base_hours = estimate_effort_hours(size, unit, multipliers) * team_overhead(team_size)
    resid = np.array(json.load(open("models/residuals.json"))["kloc" if unit == "kloc" else "fp"])

    hours = base_hours * np.exp(rng.choice(resid, n))          # effort error from the models
    avg_rate = sum(share * rates["roles"][r] for r, share in ROLE_SPLIT.items())
    rate_noise = rng.triangular(0.9, 1.0, 1.2, n)               # hourly rates vary -10% to +20%
    cloud_noise = rng.triangular(0.8, 1.0, 1.5, n)              # cloud cost varies -20% to +50%
    months = hours / (team_size * HOURS_PER_PM)

    total = (hours * avg_rate * rate_noise
             + rates["cloud_per_month"] * months * cloud_noise
             + rates["tooling_per_project"] + rates["other_per_project"])

    out = {k: round(float(np.percentile(total, p))) for k, p in
           [("P10", 10), ("P50", 50), ("P90", 90)]}
    if budget:
        out["prob_over_budget_%"] = round(float((total > budget).mean() * 100), 1)
    return out

if __name__ == "__main__":
    print("300 FP, team of 4, budget 25 lakh:")
    print(simulate(300, "fp", team_size=4, budget=2500000))
    print("50 KLOC, team of 5:")
    print(simulate(50, "kloc", team_size=5))