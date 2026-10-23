import io
import pandas as pd
from src.costing.cost_engine import ROLE_SPLIT

def build_excel(est, sim, scenarios, similar, inputs):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame(list(inputs.items()), columns=["input", "value"]).to_excel(
            w, sheet_name="Inputs", index=False)
        summary = {
            "effort_hours": est["effort_hours"],
            "duration_months": est["duration_months"],
            "best_guess_total_cost": est["total_cost"],
            "personnel_cost": est["personnel_cost"],
            "cloud_cost": est["cloud_cost"],
            "tooling_cost": est["tooling_cost"],
            "other_cost": est["other_cost"],
            **sim,
        }
        pd.DataFrame(list(summary.items()), columns=["item", "value"]).to_excel(
            w, sheet_name="Summary", index=False)
        pd.DataFrame(est["roles"]).T.to_excel(w, sheet_name="Roles")
        scenarios.to_excel(w, sheet_name="Scenarios", index=False)
        similar.to_excel(w, sheet_name="Similar projects", index=False)
    return buf.getvalue()
WEIGHT = {"low": 1, "medium": 2, "high": 3}

def task_table(tasks, total_hours, rates):
    weights = [WEIGHT[t.complexity] for t in tasks]
    present = {t.role for t in tasks}
    share_total = sum(ROLE_SPLIT[r] for r in present)
    role_weight = {r: sum(w for t, w in zip(tasks, weights) if t.role == r)
                   for r in present}
    rows = []
    for t, w in zip(tasks, weights):
        role_hours = total_hours * ROLE_SPLIT[t.role] / share_total
        h = role_hours * w / role_weight[t.role]
        rows.append({"Task": t.name, "Role": t.role, "Complexity": t.complexity,
                     "Hours": max(1, round(h)),
                     "Cost": round(h * rates["roles"][t.role])})
    return pd.DataFrame(rows)

def jira_csv(tbl):
    out = pd.DataFrame({
        "Summary": tbl["Task"],
        "Issue Type": "Task",
        "Labels": tbl["Role"],
        "Description": "Complexity: " + tbl["Complexity"],
        "Original Estimate": tbl["Hours"].astype(str) + "h",
    })
    return out.to_csv(index=False).encode("utf-8")

def msproject_csv(tbl):
    out = pd.DataFrame({
        "Name": tbl["Task"],
        "Work": tbl["Hours"].astype(str) + " hrs",
        "Resource Names": tbl["Role"],
        "Cost": tbl["Cost"],
        "Notes": "Complexity: " + tbl["Complexity"],
    })
    return out.to_csv(index=False).encode("utf-8")