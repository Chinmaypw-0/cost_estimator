import pandas as pd
from src.costing.cost_engine import cost_estimate
from src.uncertainty.monte_carlo import simulate

def run_scenarios(size, unit, base_team, budget=None):
    cases = {
        "base": (size, base_team),
        "scope +20%": (size * 1.2, base_team),
        "scope -20%": (size * 0.8, base_team),
        "team +2": (size, base_team + 2),
        "team -1": (size, max(1, base_team - 1)),
    }
    rows = []
    for name, (s, t) in cases.items():
        sim = simulate(s, unit, team_size=t, budget=budget)
        months = cost_estimate(s, unit, team_size=t)["duration_months"]
        rows.append({"scenario": name, "team": t, "months": months, **sim})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    print(run_scenarios(300, "fp", base_team=4, budget=2500000).to_string(index=False))