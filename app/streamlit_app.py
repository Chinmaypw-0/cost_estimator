import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
from src.analogy.retrieve import similar_projects
from src.costing.cost_engine import cost_estimate
from src.data.schema import MULTIPLIERS
from src.uncertainty.monte_carlo import simulate
from src.uncertainty.scenarios import run_scenarios

st.set_page_config(page_title="AI Cost Estimator", layout="wide")
st.title("AI-Based Software Project Cost Estimator")

def inr(x):
    return f"₹{x:,.0f}"

with st.sidebar:
    st.header("Project inputs")
    label = st.radio("Size is measured in", ["Function points", "KLOC (thousand lines of code)"])
    unit = "fp" if label.startswith("Function") else "kloc"
    size = st.number_input("Project size", min_value=1.0,
                           value=300.0 if unit == "fp" else 50.0,
                           step=10.0, key=f"size_{unit}")
    team = st.slider("Team size", 1, 20, 4)
    budget = st.number_input("Budget (₹)", min_value=0, value=2500000, step=100000)
    mult = {}
    if unit == "kloc":
        with st.expander("COCOMO factors (optional)"):
            for m in ["rely", "cplx", "acap"]:
                opts = list(MULTIPLIERS[m])
                choice = st.selectbox(m, opts, index=opts.index("n"))
                mult[m] = MULTIPLIERS[m][choice]
    go = st.button("Estimate", type="primary")

if go:
    est = cost_estimate(size, unit, team_size=team, multipliers=mult or None)
    sim = simulate(size, unit, team_size=team, multipliers=mult or None,
                   budget=budget or None)

    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated effort (hours)", f"{est['effort_hours']:,}")
    c2.metric("Duration (months)", est["duration_months"])
    c3.metric("Best-guess total cost", inr(est["total_cost"]))

    st.subheader("Confidence range")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("P10 (best case)", inr(sim["P10"]))
    r2.metric("P50 (likely)", inr(sim["P50"]))
    r3.metric("P90 (pessimistic)", inr(sim["P90"]))
    if "prob_over_budget_%" in sim:
        r4.metric("Chance of going over budget", f"{sim['prob_over_budget_%']}%")

    left, right = st.columns(2)
    with left:
        st.subheader("Cost by role")
        st.dataframe(pd.DataFrame(est["roles"]).T)
    with right:
        st.subheader("Cost by category")
        st.bar_chart(pd.Series({"Personnel": est["personnel_cost"],
                                "Cloud": est["cloud_cost"],
                                "Tooling": est["tooling_cost"],
                                "Other": est["other_cost"]}))

    st.subheader("What-if scenarios")
    st.dataframe(run_scenarios(size, unit, team, budget=budget or None))
    st.caption("Scenarios use nominal COCOMO factors.")

    st.subheader("Similar past projects")
    st.dataframe(similar_projects(size, unit))
else:
    st.info("Enter the project details on the left and click Estimate.")