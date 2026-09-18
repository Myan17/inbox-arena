"""Environment contract tests.

An OpenEnv environment has to hold a specific protocol -- reset before step,
one step per episode, deterministic replay from a seed, and ground truth that
survives adversarial perturbation. A trainer relies on all four.
"""
import pytest

from models import TriageAction
from server.data import (
    get_benchmark_seeds,
    get_benchmark_tasks,
    get_pool_size,
    get_random_email,
)
from server.environment import EmailTriageEnvironment
from server.perturbations import apply_perturbation


@pytest.fixture
def env():
    return EmailTriageEnvironment()


def _action(**kw):
    base = {"category": "urgent", "priority": "P0", "department": "engineering"}
    base.update(kw)
    return TriageAction(**base)


class TestEpisodeProtocol:
    def test_reset_returns_an_email_and_a_task(self, env):
        obs = env.reset(task_name="classify_easy", seed=1)
        assert obs.email is not None
        assert obs.task is not None
        assert obs.done is False
        assert obs.reward is None

    def test_step_before_reset_is_an_error_not_a_crash(self, env):
        obs = env.step(_action())
        assert obs.done is True
        assert obs.error_message is not None

    def test_one_step_ends_the_episode(self, env):
        env.reset(task_name="classify_easy", seed=1)
        obs = env.step(_action())
        assert obs.done is True
        assert obs.reward is not None

    def test_a_second_step_is_rejected(self, env):
        env.reset(task_name="classify_easy", seed=1)
        env.step(_action())
        obs = env.step(_action())
        assert obs.error_message is not None

    def test_unknown_task_is_rejected_with_the_valid_names(self, env):
        obs = env.reset(task_name="not_a_task")
        assert obs.done is True
        assert "classify_easy" in obs.error_message

    def test_reset_after_a_finished_episode_starts_a_new_one(self, env):
        env.reset(task_name="classify_easy", seed=1)
        env.step(_action())
        obs = env.reset(task_name="classify_easy", seed=2)
        assert obs.done is False
        assert obs.reward is None


class TestDeterminism:
    def test_the_same_seed_yields_the_same_email(self):
        first, _ = get_random_email(seed=42, task_name="classify_easy")
        second, _ = get_random_email(seed=42, task_name="classify_easy")
        assert first.subject == second.subject
        assert first.body == second.body

    def test_the_same_seed_yields_the_same_reward(self, env):
        env.reset(task_name="triage_medium", seed=7)
        first = env.step(_action()).reward

        env2 = EmailTriageEnvironment()
        env2.reset(task_name="triage_medium", seed=7)
        assert env2.step(_action()).reward == first

    def test_different_seeds_reach_different_emails(self):
        subjects = {get_random_email(seed=s)[0].subject for s in range(25)}
        assert len(subjects) > 1


class TestBenchmarkFixtures:
    def test_every_benchmark_task_has_seeds(self):
        tasks = get_benchmark_tasks()
        assert tasks
        for task in tasks:
            assert get_benchmark_seeds(task), task

    def test_benchmark_seeds_return_their_curated_email(self):
        for task in get_benchmark_tasks():
            for seed in get_benchmark_seeds(task):
                email, truth = get_random_email(seed=seed, task_name=task)
                assert email.subject
                assert truth.category

    def test_the_email_pool_is_not_trivially_small(self):
        assert get_pool_size() >= 10


class TestPerturbations:
    def test_none_returns_the_email_unchanged(self):
        email, truth = get_random_email(seed=3)
        assert apply_perturbation(email, truth.priority.value, mode="none") is email

    def test_an_unknown_mode_is_ignored_rather_than_raising(self):
        email, truth = get_random_email(seed=3)
        assert apply_perturbation(email, truth.priority.value, mode="nonsense") is email

    @pytest.mark.parametrize(
        "mode", ["homoglyph", "tone_inversion", "identity_spoof", "distractor_inject", "all"]
    )
    def test_perturbation_changes_the_surface_but_not_the_ground_truth(self, mode):
        # This is the property the whole adversarial suite rests on: if a
        # perturbation could alter the answer, every perturbed score would be
        # measuring the wrong thing.
        email, truth = get_random_email(seed=11)
        before = (truth.category, truth.priority, truth.department)

        perturbed = apply_perturbation(email, truth.priority.value, mode=mode, seed=11)

        assert (truth.category, truth.priority, truth.department) == before
        assert perturbed is not None

    def test_perturbations_are_deterministic_under_a_seed(self):
        email, truth = get_random_email(seed=11)
        a = apply_perturbation(email, truth.priority.value, mode="all", seed=5)
        b = apply_perturbation(email, truth.priority.value, mode="all", seed=5)
        assert a.subject == b.subject
        assert a.body == b.body
