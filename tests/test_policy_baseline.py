"""Pin the policy-ladder benchmark.

These assertions encode what the reward function must do to be usable as a
training signal: rank an attentive agent above a fluent-but-empty one, and both
above a constant guesser. If a grader change breaks that ordering, CI fails
here rather than silently producing meaningless RL results.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

from policy_baseline import (  # noqa: E402
    TASKS,
    evaluate,
    policy_attentive,
    policy_constant,
    policy_fields_only,
    policy_keyword,
    policy_random,
    policy_slop,
)

EPISODES = 60


def _mean(policy, task):
    scores = evaluate(policy, task, EPISODES)
    return sum(scores) / len(scores)


@pytest.mark.parametrize("task", TASKS)
def test_attentive_beats_constant_by_a_wide_margin(task):
    assert _mean(policy_attentive, task) - _mean(policy_constant, task) > 0.5


@pytest.mark.parametrize("task", TASKS)
def test_a_heuristic_beats_random_guessing(task):
    assert _mean(policy_keyword, task) > _mean(policy_random, task)


def test_the_hard_task_rewards_reading_the_email_over_sounding_polite():
    # The grader's stated purpose. A slop draft is fluent and contentless; an
    # attentive draft cites the entities. Both have perfect structured fields,
    # so the whole gap is the response quality rubric.
    slop = _mean(policy_slop, "full_triage_hard")
    attentive = _mean(policy_attentive, "full_triage_hard")
    assert attentive - slop > 0.25


def test_a_draft_is_worth_more_than_no_draft_on_the_hard_task():
    assert _mean(policy_slop, "full_triage_hard") > _mean(policy_fields_only, "full_triage_hard")


@pytest.mark.parametrize("task", ["classify_easy", "triage_medium"])
def test_perfect_fields_saturate_the_non_draft_tasks(task):
    # Easy and medium grade structured fields only, so a policy handed the
    # ground truth must approach 1.0. Anything less means a grader bug.
    assert _mean(policy_fields_only, task) > 0.99


def test_published_means_have_not_drifted():
    # The figures quoted in the README.
    assert _mean(policy_constant, "full_triage_hard") == pytest.approx(0.330, abs=0.01)
    assert _mean(policy_keyword, "full_triage_hard") == pytest.approx(0.397, abs=0.01)
    assert _mean(policy_slop, "full_triage_hard") == pytest.approx(0.617, abs=0.01)
    assert _mean(policy_attentive, "full_triage_hard") == pytest.approx(0.928, abs=0.01)


def test_the_benchmark_is_deterministic():
    first = evaluate(policy_keyword, "triage_medium", 20)
    second = evaluate(policy_keyword, "triage_medium", 20)
    assert first == second
