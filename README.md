AI-Based Software Project Cost Estimator 

Describe a software project in plain English and get an estimate of effort, duration and
cost, with a confidence range, what-if scenarios, similar past projects, and exports
to Excel, Jira and MS Project.

## How it works
1. Describe the project. Gemini reads the text and returns a project type, a size
   estimate, a team size and a task list. If it is not confident (below 70%), the app asks
   the user to confirm the type. The user can edit everything before estimating.
2. Effort model. Function-point projects use a log-log size model; KLOC projects use a
   COCOMO-style model (size plus 15 effort multipliers).
3. Cost engine. Effort is split across roles, priced with the editable rate card in
   `config/rates.yaml`, and cloud, tooling and other costs are added.
4. Monte-Carlo simulation (10,000 runs) gives P10 / P50 / P90 and the chance of going
   over a budget.
5. Scenarios (scope +/-20%, team size changes), similar past projects, and
   exports (Excel, Jira CSV, MS Project CSV).

## Data
NASA93, COCOMO81, Desharnais, Albrecht, Kemerer, Maxwell and China (677 function-point
and 156 KLOC projects in total). Effort is converted to person-hours. Sources and licences
are listed in `data/SOURCES.md`. Raw data is not stored in the repository. ISBSG is paid
and was not used.

## Results (cross-validation on held-out projects)
| Family | Model | PRED(25) | MdMRE |
|---|---|---|---|
| KLOC | Size only | 28.9% | 0.48 |
| KLOC | COCOMO-style (size + 15 factors) | 45.5% | 0.28 |
| KLOC | Ridge / Random Forest / Gradient Boosting | 42.3% / 34.0% / 35.3% | 0.29 / 0.36 / 0.37 |
| Function points | Log-log size model | 19.7% | 0.56 |

- The ML models did not beat the simple COCOMO-style model on this small data.
- Domain-specific models did not beat one general model.
- Interval calibration: the 80% range (P10 to P90) caught 81.4% of unseen
  function-point projects and 79.5% of KLOC projects.
- Bias: real effort was above the estimate in 52.4% (function points) and 58.3% (KLOC)
  of cases, so KLOC estimates lean slightly low. Read the P50 to P90 range, not only the
  best guess.

## Run it
```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash
pip install -r requirements.txt
```
1. Put the datasets in `data/raw/<name>/` (see `data/SOURCES.md`).
2. Create `.env` with `GEMINI_API_KEY=your_key` (and optionally `GEMINI_MODEL=...`).
3. Build the data and models:
```bash
python -m src.data.clean
python -m src.models.final_models
```
4. Start the app and run the tests:
```bash
streamlit run app/streamlit_app.py
pytest
```

## Limitations
- Cloud, tooling and "other" costs, hourly rates, the role split (55/20/10/10/5) and the
  5% team overhead are editable assumptions, not live prices or learned values.
- Function-point estimates are very uncertain (only about 20% land within 25% of actual).
- The LLM's size guess is the weakest link, so the user must review it.
- Per-task hours in the Jira and MS Project exports are planning splits, not real per-task
  estimates, and the CSV imports are untested in real Jira or MS Project.
- Trained on public research datasets, not on any company's own projects.
- The free Gemini tier allows only about 20 requests per day.

## Project structure
`src/data` cleaning, `src/models` training and validation, `src/costing` cost engine,
`src/uncertainty` Monte-Carlo and scenarios, `src/analogy` similar projects,
`src/llm` description reader, `src/export.py` exports, `app/` Streamlit app, `tests/`.