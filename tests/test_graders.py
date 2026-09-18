"""Grader tests.

The graders are the environment's reward function: every RL result produced
against InboxArena is only as trustworthy as these are. They are pure and
deterministic, so they are asserted exactly.
"""
import pytest

from models import Department, EmailCategory, GroundTruth, Priority, TriageAction
from server.graders import grade, grade_easy, grade_hard, grade_medium


@pytest.fixture
def truth():
    return GroundTruth(
        category=EmailCategory.URGENT,
        priority=Priority.P0,
        department=Department.ENGINEERING,
        expected_response_keywords=["bridge", "eta"],
        expected_response_tone="urgent",
        expected_entities=["INC-PAY-4471"],
    )


def _action(**kw):
    base = {"category": "urgent", "priority": "P0", "department": "engineering"}
    base.update(kw)
    return TriageAction(**base)


class TestEasyGrader:
    def test_correct_category_scores_full_marks(self, truth):
        score, _ = grade_easy(_action(), truth)
        assert score == 1.0

    def test_wrong_category_scores_zero(self, truth):
        score, feedback = grade_easy(_action(category="spam"), truth)
        assert score == 0.0
        assert "urgent" in feedback

    def test_category_matching_ignores_case_and_whitespace(self, truth):
        assert grade_easy(_action(category="  URGENT "), truth)[0] == 1.0


class TestMediumGrader:
    def test_all_three_fields_correct_scores_one(self, truth):
        assert grade_medium(_action(), truth)[0] == 1.0

    @pytest.mark.parametrize(
        "override,expected",
        [
            ({"category": "spam"}, 0.60),
            ({"priority": "P3"}, 0.70),
            ({"department": "finance"}, 0.70),
        ],
    )
    def test_partial_credit_matches_the_published_weights(self, truth, override, expected):
        # 0.40 category + 0.30 priority + 0.30 department.
        assert grade_medium(_action(**override), truth)[0] == pytest.approx(expected)

    def test_everything_wrong_scores_zero(self, truth):
        action = _action(category="spam", priority="P3", department="finance")
        assert grade_medium(action, truth)[0] == 0.0


class TestHardGrader:
    def test_a_draft_citing_concrete_details_beats_generic_slop(self, truth):
        # This is the grader's central design claim, so it is asserted directly.
        good = _action(response_draft=(
            "Joining the INC-PAY-4471 bridge now, ETA 2 minutes. "
            "Pulling the payment service logs on the way in."
        ))
        slop = _action(response_draft=(
            "Thank you for your email. I will look into this and get back to you "
            "as soon as possible. Please let me know if you have any questions."
        ))
        good_score, _ = grade_hard(good, truth)
        slop_score, _ = grade_hard(slop, truth)
        assert good_score > slop_score

    def test_an_empty_draft_still_earns_the_structured_fields(self, truth):
        # 0.15 + 0.10 + 0.10 for category/priority/department.
        score, _ = grade_hard(_action(response_draft=""), truth)
        assert score >= 0.35 - 1e-9

    def test_citing_the_required_entity_is_rewarded(self, truth):
        without = grade_hard(_action(response_draft="Joining bridge now, ETA 2 min."), truth)[0]
        with_entity = grade_hard(_action(response_draft="Joining INC-PAY-4471 bridge now, ETA 2 min."), truth)[0]
        assert with_entity > without

    def test_score_never_leaves_the_reward_range(self, truth):
        # Observation.reward is declared gt=0.0 lt=1.0, so a grader returning
        # exactly 0 or above 1.05 would make the response unserialisable.
        for draft in ["", "x", "INC-PAY-4471 bridge eta " * 50]:
            for conf in [None, 0.0, 0.5, 1.0]:
                score, _ = grade_hard(_action(response_draft=draft, confidence=conf), truth)
                assert 0.0 <= score <= 1.05


class TestConfidenceCalibration:
    def test_well_calibrated_confidence_earns_the_bonus(self, truth):
        # Claims 1.0, scores 1.0 -> Brier 0 -> +0.05.
        assert grade_easy(_action(confidence=1.0), truth)[0] == pytest.approx(1.05)

    def test_overconfidence_on_a_wrong_answer_is_penalised(self, truth):
        score, _ = grade_easy(_action(category="spam", confidence=1.0), truth)
        assert score < 0.0 + 1e-9 or score == 0.0

    def test_omitting_confidence_is_neutral(self, truth):
        assert grade_easy(_action(confidence=None), truth)[0] == 1.0

    def test_bonus_is_capped_and_penalty_is_floored(self, truth):
        best = grade_medium(_action(confidence=1.0), truth)[0]
        worst = grade_medium(_action(category="spam", priority="P3", department="finance", confidence=1.0), truth)[0]
        assert best <= 1.05
        assert worst >= 0.0


class TestGradeDispatch:
    @pytest.mark.parametrize("task", ["classify_easy", "triage_medium", "full_triage_hard"])
    def test_every_task_name_dispatches(self, truth, task):
        score, feedback = grade(task, _action(response_draft="INC-PAY-4471 bridge, ETA 2 min."), truth)
        assert 0.0 <= score <= 1.05
        assert feedback

    def test_grading_is_deterministic(self, truth):
        action = _action(response_draft="Joining INC-PAY-4471 bridge, ETA 2 minutes.", confidence=0.8)
        first = grade("full_triage_hard", action, truth)
        for _ in range(5):
            assert grade("full_triage_hard", action, truth) == first
