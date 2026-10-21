import io
import pandas as pd

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