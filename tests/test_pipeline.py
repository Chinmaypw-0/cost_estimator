from types import SimpleNamespace

import pytest

from src.analogy.retrieve import similar_projects
from src.costing.cost_engine import ROLE_SPLIT, cost_estimate, load_rates
from src.export import task_table
from src.uncertainty.monte_carlo import simulate


def test_costs_add_up():
    est = cost_estimate(300, "fp", team_size=4)
    assert sum(r["cost"] for r in est["roles"].values()) == est["personnel_cost"]
    parts = (est["personnel_cost"] + est["cloud_cost"]
             + est["tooling_cost"] + est["other_cost"])
    assert parts == pytest.approx(est["total_cost"], abs=2)


def test_bigger_projects_cost_more():
    assert cost_estimate(500, "fp")["total_cost"] > cost_estimate(100, "fp")["total_cost"]
    assert cost_estimate(100, "kloc")["total_cost"] > cost_estimate(20, "kloc")["total_cost"]


def test_bigger_team_costs_more_but_finishes_sooner():
    small = cost_estimate(300, "fp", team_size=3)
    big = cost_estimate(300, "fp", team_size=8)
    assert big["total_cost"] > small["total_cost"]
    assert big["duration_months"] < small["duration_months"]


def test_range_is_ordered():
    s = simulate(300, "fp", team_size=4, budget=2500000)
    assert s["P10"] < s["P50"] < s["P90"]
    assert 0 <= s["prob_over_budget_%"] <= 100


def test_task_hours_follow_role_split():
    tasks = [
        SimpleNamespace(name="a", role="developer", complexity="medium"),
        SimpleNamespace(name="b", role="developer", complexity="low"),
        SimpleNamespace(name="c", role="tester", complexity="medium"),
        SimpleNamespace(name="d", role="designer", complexity="low"),
        SimpleNamespace(name="e", role="devops", complexity="low"),
        SimpleNamespace(name="f", role="project_manager", complexity="low"),
    ]
    tbl = task_table(tasks, 1000, load_rates())
    by_role = tbl.groupby("Role")["Hours"].sum()
    for role, share in ROLE_SPLIT.items():
        assert by_role[role] == pytest.approx(1000 * share, abs=3)


def test_similar_projects_are_close_in_size():
    sim = similar_projects(300, "fp", k=3)
    assert len(sim) == 3
    assert (abs(sim["size_value"] - 300) < 60).all()