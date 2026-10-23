import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
from src.analogy.retrieve import similar_projects
from src.costing.cost_engine import cost_estimate, load_rates
from src.llm.decompose import decompose
from src.uncertainty.monte_carlo import simulate
from src.uncertainty.scenarios import run_scenarios
from src.export import build_excel, task_table, jira_csv, msproject_csv
from src.db import save_estimate, list_estimates, set_actual, kpis

DOMAINS = ["business_software", "web_mobile", "data_ai",
           "embedded_iot", "engineering_scientific", "other"]
UNIT_FOR = {"embedded_iot": "kloc", "engineering_scientific": "kloc"}   # others use fp
CONF_THRESHOLD = 0.7

st.set_page_config(page_title="AI Cost Estimator", layout="wide")
st.title("AI-Based Software Project Cost Estimator")

def inr(x):
    return f"₹{x:,.0f}"

def show_results(size, unit, team, budget,tasks):
    est = cost_estimate(size, unit, team_size=team)
    sim = simulate(size, unit, team_size=team, budget=budget or None)

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
        st.bar_chart(pd.Series({"Personnel": est["personnel_cost"], "Cloud": est["cloud_cost"],
                                "Tooling": est["tooling_cost"], "Other": est["other_cost"]}))

    scen = run_scenarios(size, unit, team, budget=budget or None)
    sim_proj = similar_projects(size, unit)
    st.subheader("What-if scenarios")
    st.dataframe(scen)
    st.subheader("Similar past projects")
    st.dataframe(sim_proj)

    inputs = {"size": size, "size_unit": unit, "team_size": team, "budget": budget}
    st.download_button("Download estimate (Excel)",
                       build_excel(est, sim, scen, sim_proj, inputs),
                       file_name="cost_estimate.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    tbl = task_table(tasks, est["effort_hours"], load_rates())
    st.subheader("Export tasks")
    st.dataframe(tbl)
    b1, b2 = st.columns(2)
    b1.download_button("Download for Jira (CSV)", jira_csv(tbl),
                       file_name="jira_tasks.csv", mime="text/csv")
    b2.download_button("Download for MS Project (CSV)", msproject_csv(tbl),
                       file_name="msproject_tasks.csv", mime="text/csv")
st.header("1. Describe your project")
desc = st.text_area("Describe the project in plain English", height=120,
                    placeholder="e.g. An inventory web app with login, stock alerts and sales reports")

if st.button("Analyze description"):
    if not desc.strip():
        st.warning("Please type a description first.")
    else:
        with st.spinner("Reading your description..."):
            try:
                st.session_state["bd"] = decompose(desc)
                st.session_state["aid"] = st.session_state.get("aid", 0) + 1
                st.session_state["show"] = False
            except Exception as e:
                st.error(f"Could not analyze the description: {e}")

bd = st.session_state.get("bd")
if bd:
    aid = st.session_state["aid"]
    st.header("2. Review and edit")

    if bd.domain_confidence < CONF_THRESHOLD or bd.domain == "other":
        st.warning(f"I'm not sure about the project type (confidence {bd.domain_confidence:.0%}). "
                   "Please confirm it below. 'other' uses the general model.")
    else:
        st.success(f"Detected: {bd.domain} (confidence {bd.domain_confidence:.0%}). You can change it.")

    col1, col2, col3 = st.columns(3)
    domain = col1.selectbox("Project type", DOMAINS, index=DOMAINS.index(bd.domain), key=f"d{aid}")
    unit = UNIT_FOR.get(domain, "fp")
    if unit != bd.size_unit:
        st.warning(f"The size was estimated in {bd.size_unit}, but this type uses {unit}. "
                   "Please enter a new size.")
        default_size = 300.0 if unit == "fp" else 50.0
    else:
        default_size = float(bd.size_value)
    size = col2.number_input(f"Size ({'function points' if unit == 'fp' else 'KLOC'})",
                             min_value=1.0, value=default_size, step=10.0, key=f"s{aid}{unit}")
    team = col3.slider("Team size", 1, 20, int(bd.suggested_team_size), key=f"t{aid}")
    st.caption(f"Size reasoning: {bd.size_reasoning}")
    budget = st.number_input("Budget (₹)", min_value=0, value=2500000, step=100000)

    with st.expander("Task breakdown (for display; cost uses a fixed role split)"):
        st.dataframe(pd.DataFrame([t.model_dump() for t in bd.tasks]))

    if st.button("Estimate", type="primary"):
        st.session_state["show"] = True
        _est = cost_estimate(size, unit, team_size=team)
        _sim = simulate(size, unit, team_size=team, budget=budget or None)
        save_estimate(desc, domain, size, unit, team, budget, _est, _sim)
    if st.session_state.get("show"):
        st.header("3. Results")
        show_results(size, unit, team, budget, bd.tasks)
else:
    st.info("Describe your project above and click Analyze description.")
st.divider()
with st.expander("Saved estimates"):
    hist = list_estimates()
    st.dataframe(hist)
    if not hist.empty:
        st.markdown("**Record the real cost of a finished project**")
        c1, c2 = st.columns(2)
        pick = c1.selectbox("Estimate id", hist["id"].tolist())
        actual = c2.number_input("Actual total cost (₹)", min_value=0, step=10000)
        if st.button("Save actual cost"):
            set_actual(int(pick), float(actual))
            st.rerun()
    k = kpis()
    if k:
        st.markdown("**Accuracy so far**")
        st.table(pd.DataFrame([(a, str(b)) for a, b in k.items()],
                              columns=["KPI", "value"]))