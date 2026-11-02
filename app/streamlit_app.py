import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.db import save_estimate, list_estimates, set_actual, kpis, delete_estimate

import pandas as pd
import hashlib
import os
import numpy as np
import streamlit as st
from src.analogy.retrieve import similar_projects
from src.costing.cost_engine import cost_estimate, load_rates
from src.llm.decompose import decompose
from src.uncertainty.monte_carlo import simulate
from src.uncertainty.scenarios import run_scenarios
from src.export import build_excel, task_table, jira_csv, msproject_csv
from src.db import save_estimate, list_estimates, set_actual, kpis, delete_estimate

DOMAINS = ["business_software", "web_mobile", "data_ai",
           "embedded_iot", "engineering_scientific", "other"]
UNIT_FOR = {"embedded_iot": "kloc", "engineering_scientific": "kloc"}   # others use fp
CONF_THRESHOLD = 0.7

st.set_page_config(page_title="AI Cost Estimator", page_icon="📊", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
[data-testid="stMetric"] {
    background: var(--secondary-background-color, rgba(127,127,127,.08));
    border: 1px solid rgba(127,127,127,.28);
    border-radius: 12px; padding: 14px 16px;
}
[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
h1 {font-weight: 700; letter-spacing: -0.02em;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("AI-Based Software Project Cost Estimator")
st.caption("Describe a project in plain English, or upload a document, and get effort, "
           "cost and a confidence range, with scenarios and exports.")


def inr(x):
    if x >= 1e7:
        return f"₹{x / 1e7:.2f} Cr"
    if x >= 1e5:
        return f"₹{x / 1e5:.1f} L"
    return f"₹{x:,.0f}"

PROMPT_VERSION = "1"   # change this number whenever you edit the prompt in decompose.py


def _key(desc, doc):
    h = hashlib.sha256()
    h.update(" ".join(desc.split()).lower().encode())
    if doc:
        h.update(doc[0].encode())
        h.update(doc[1])
    h.update((os.getenv("GEMINI_MODEL", "") + PROMPT_VERSION).encode())
    return h.hexdigest()


@st.cache_data(persist="disk", show_spinner=False)
def cached_decompose(key, _desc, _doc):
    return decompose(_desc, _doc)

def range_bar(lo, mid, hi, budget=0):
    top = max(hi, budget) * 1.05
    pct = lambda v: v / top * 100
    color = "var(--primary-color, #0F766E)"
    budget_line = (f'<div style="position:absolute;left:{pct(budget):.1f}%;top:-6px;bottom:-6px;'
                   f'width:2px;background:#E5484D"></div>') if budget else ""
    html = (
        '<div style="margin:10px 0 6px;"><div style="position:relative;height:14px;'
        'border-radius:7px;background:rgba(127,127,127,.2);">'
        f'<div style="position:absolute;top:0;bottom:0;left:{pct(lo):.1f}%;'
        f'width:{pct(hi) - pct(lo):.1f}%;border-radius:7px;background:{color};opacity:.45;"></div>'
        f'<div style="position:absolute;top:-5px;bottom:-5px;left:{pct(mid):.1f}%;'
        f'width:4px;border-radius:2px;background:{color};"></div>'
        f'{budget_line}</div></div>')
    st.markdown(html, unsafe_allow_html=True)


# Heavy calculations are cached, so if at all tabs are switched it wont recomput...
@st.cache_data(show_spinner=False)
def get_est(size, unit, team):
    return cost_estimate(size, unit, team_size=team)

@st.cache_data(show_spinner=False)
def get_sim(size, unit, team, budget):
    return simulate(size, unit, team_size=team, budget=budget or None)

@st.cache_data(show_spinner=False)
def get_scen(size, unit, team, budget):
    return run_scenarios(size, unit, team, budget=budget or None)

@st.cache_data(show_spinner=False)
def get_similar(size, unit):
    return similar_projects(size, unit)


tab_new, tab_what, tab_sim, tab_hist, tab_exp = st.tabs(
    ["New estimate", "What-if", "Similar projects", "History", "Export"])
NEED = "Make an estimate first on the New estimate tab."

#New estimate------
with tab_new:
    st.header("1. Describe your project")
    busy = st.session_state.get("busy", False)

    desc = st.text_area("Describe the project in plain English", height=120, key="desc_in",
                        disabled=busy,
                        placeholder="e.g. An inventory web app with login, stock alerts and sales reports")
    upload = st.file_uploader(
        "Or upload a requirements document, PDF or screenshot",
        type=["pdf", "png", "jpg", "jpeg", "docx", "txt", "md"],
        key="upload_in", disabled=busy)

    st.button("Analyze description", type="primary", disabled=busy,
              on_click=lambda: st.session_state.update(busy=True, msg=None))

    if busy:
        if not desc.strip() and upload is None:
            st.session_state["msg"] = ("warning", "Please type a description or upload a file first.")
        elif upload is not None and upload.size > 10 * 1024 * 1024:
            st.session_state["msg"] = ("error", "That file is over 10 MB. Please upload a smaller one.")
        else:
            with st.spinner("Reading your project..."):
                try:
                    doc = (upload.name, upload.getvalue()) if upload else None
                    st.session_state["bd"] = cached_decompose(_key(desc, doc), desc, doc)
                    st.session_state["aid"] = st.session_state.get("aid", 0) + 1
                    st.session_state["res"] = None
                except Exception as e:
                    st.session_state["msg"] = ("error", f"Could not analyze the description: {e}")
        st.session_state["busy"] = False
        st.rerun()

    if st.session_state.get("msg"):
        kind, text = st.session_state["msg"]
        getattr(st, kind)(text)

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
        budget = st.number_input("Budget (₹), optional. Leave 0 if you have none",
                         min_value=0, value=0, step=100000)

        with st.expander("Task breakdown (for display; cost uses a fixed role split)"):
            st.dataframe(pd.DataFrame([t.model_dump() for t in bd.tasks]))

        if st.button("Estimate", type="primary"):
            _est = get_est(size, unit, team)
            _sim = get_sim(size, unit, team, budget)
            save_estimate(desc, domain, size, unit, team, budget, _est, _sim)
            st.session_state["res"] = {"size": size, "unit": unit, "team": team,
                                       "budget": budget, "tasks": bd.tasks}

        res = st.session_state.get("res")
        if res:
            st.header("3. Results")
            st.caption("These numbers are for the inputs used the last time you clicked Estimate.")
            est = get_est(res["size"], res["unit"], res["team"])
            sim = get_sim(res["size"], res["unit"], res["team"], res["budget"])

            c1, c2, c3 = st.columns(3)
            c1.metric("Estimated effort (hours)", f"{est['effort_hours']:,}")
            c2.metric("Duration (months)", est["duration_months"])
            c3.metric("Most likely total cost", inr(sim["P50"]))

            st.subheader("Likely cost range")
            range_bar(sim["P10"], sim["P50"], sim["P90"], res["budget"])
            st.caption("About 8 out of 10 times the final cost lands inside the shaded range. "
                       "The solid mark is the most likely value."
                       + (" The red line is your budget." if res["budget"] else ""))
            r1, r2, r3 = st.columns(3)
            r1.metric("Low (best case)", inr(sim["P10"]))
            r2.metric("High (worst case)", inr(sim["P90"]))
            if "prob_over_budget_%" in sim:
                r3.metric("Chance of going over budget", f"{sim['prob_over_budget_%']}%")

            st.caption(f"The role and category breakdown below uses the model's single estimate "
                       f"({inr(est['total_cost'])}). The most likely total above also includes the "
                       "simulated uncertainty, so the two differ slightly.")
            left, right = st.columns(2)
            with left:
                st.subheader("Cost by role")
                st.dataframe(pd.DataFrame(est["roles"]).T)
            with right:
                st.subheader("Cost by category")
                st.bar_chart(pd.Series({"Personnel": est["personnel_cost"], "Cloud": est["cloud_cost"],
                                        "Tooling": est["tooling_cost"], "Other": est["other_cost"]}))
    else:
        st.info("Describe your project above and click Analyze description.")

res = st.session_state.get("res")

#What-if-----
with tab_what:
    if not res:
        st.info(NEED)
    else:
        st.subheader("Try your own change")
        key = f"{res['team']}_{res['size']}"
        w1, w2 = st.columns(2)
        new_team = w1.slider("Team size", 1, 20, int(res["team"]), key=f"w_team_{key}")
        scope = w2.slider("Change in project scope (%)", -50, 100, 0, step=5, key=f"w_scope_{key}")
        new_size = max(1.0, res["size"] * (1 + scope / 100))

        base_est = get_est(res["size"], res["unit"], res["team"])
        base_sim = get_sim(res["size"], res["unit"], res["team"], res["budget"])
        new_est = get_est(new_size, res["unit"], new_team)
        new_sim = get_sim(new_size, res["unit"], new_team, res["budget"])

        def money_delta(new, old):
            d = new - old
            return f"{'+' if d >= 0 else '-'}{inr(abs(d))}"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Duration (months)", new_est["duration_months"],
                  delta=f"{new_est['duration_months'] - base_est['duration_months']:+.1f}",
                  delta_color="inverse")
        m2.metric("Low (best case)", inr(new_sim["P10"]),
                  delta=money_delta(new_sim["P10"], base_sim["P10"]), delta_color="inverse")
        m3.metric("Mid (most likely)", inr(new_sim["P50"]),
                  delta=money_delta(new_sim["P50"], base_sim["P50"]), delta_color="inverse")
        m4.metric("High (worst case)", inr(new_sim["P90"]),
                  delta=money_delta(new_sim["P90"], base_sim["P90"]), delta_color="inverse")
        st.caption("The small arrows compare with your last estimate. Green means cheaper or faster. "
                   "Only the cost engine and simulation run again, so Gemini is not called.")

        with st.expander("Preset scenarios (table)"):
            st.dataframe(get_scen(res["size"], res["unit"], res["team"], res["budget"]))

#Similar projects------
with tab_sim:
    if not res:
        st.info(NEED)
    else:
        st.subheader("From the research dataset")
        st.caption("Real past projects of about the same size.")
        st.dataframe(get_similar(res["size"], res["unit"]))

        st.subheader("From your saved estimates")
        past = list_estimates()
        if not past.empty:
            past = past[past["unit"] == res["unit"]].copy()
            if not past.empty:
                latest = past["id"].max()          # the estimate just saved
                is_same = (past["id"] == latest) & (past["size"] == res["size"]) & (past["team"] == res["team"])
                past = past[~is_same]
        if past.empty:
            st.info("No earlier estimates of this kind yet. They will appear here as you save more.")
        else:
            past["gap"] = (np.log(past["size"]) - np.log(res["size"])).abs()
            close = past.sort_values("gap").head(5)
            cols = ["id", "created", "domain", "size", "team", "best_guess", "p50", "actual_cost"]
            st.caption("Estimates you made before, closest in size first. "
                       "If a real cost was recorded, it shows in actual_cost.")
            st.dataframe(close[[c for c in cols if c in close.columns]])

# History----
with tab_hist:
    st.subheader("Saved estimates")
    hist = list_estimates()
    st.dataframe(hist)
    if not hist.empty:
        st.markdown("**Record the real cost of a finished project**")
        h1, h2 = st.columns(2)
        pick = h1.selectbox("Estimate id", hist["id"].tolist())
        actual = h2.number_input("Actual total cost (₹)", min_value=0, step=10000)
        if st.button("Save actual cost"):
            set_actual(int(pick), float(actual))
            st.rerun()
            st.markdown("**Delete a saved estimate**")

        def _delete():
            delete_estimate(int(st.session_state["del_id"]))
            st.session_state["del_sure"] = False

        d1, d2 = st.columns(2)
        d1.selectbox("Estimate to delete", hist["id"].tolist(), key="del_id")
        d2.checkbox("Yes, delete it permanently", key="del_sure")
        st.button("Delete estimate", disabled=not st.session_state.get("del_sure", False),
                  on_click=_delete)
    k = kpis()
    if k:
        st.markdown("**Accuracy so far**")
        st.table(pd.DataFrame([(a, str(b)) for a, b in k.items()],
                              columns=["KPI", "value"]))

#Export-------
with tab_exp:
    if not res:
        st.info(NEED)
    else:
        est = get_est(res["size"], res["unit"], res["team"])
        sim = get_sim(res["size"], res["unit"], res["team"], res["budget"])
        scen = get_scen(res["size"], res["unit"], res["team"], res["budget"])
        sim_proj = get_similar(res["size"], res["unit"])
        inputs = {"size": res["size"], "size_unit": res["unit"],
                  "team_size": res["team"], "budget": res["budget"]}
        st.download_button("Download estimate (Excel)",
                           build_excel(est, sim, scen, sim_proj, inputs),
                           file_name="cost_estimate.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        tbl = task_table(res["tasks"], est["effort_hours"], load_rates())
        st.subheader("Export tasks")
        st.dataframe(tbl)
        b1, b2 = st.columns(2)
        b1.download_button("Download for Jira (CSV)", jira_csv(tbl),
                           file_name="jira_tasks.csv", mime="text/csv")
        b2.download_button("Download for MS Project (CSV)", msproject_csv(tbl),
                           file_name="msproject_tasks.csv", mime="text/csv")