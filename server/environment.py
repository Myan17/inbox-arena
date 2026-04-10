"""
Email Triage Environment — core OpenEnv-compliant implementation.

Implements the three required methods:
  - reset(task_name)  → initial Observation
  - step(action)      → Observation with reward, done, feedback
  - state()           → current State metadata
"""

from __future__ import annotations

import uuid
from typing import Optional

from models import (
    Department,
    EmailCategory,
    GroundTruth,
    Observation,
    Priority,
    State,
    TaskInfo,
    TriageAction,
)
from server.data import get_random_email
from server.graders import grade
from server.perturbations import apply_perturbation, VALID_PERTURBATIONS


# ── Task Definitions ─────────────────────────────────────────────────────────

TASK_DEFINITIONS = {
    "classify_easy": TaskInfo(
        task_name="classify_easy",
        difficulty="easy",
        instructions=(
            "Classify this email into one of the following categories: "
            "spam, urgent, routine, newsletter, personal. "
            "Set the 'category' field in your action."
        ),
        required_fields=["category"],
        available_categories=[c.value for c in EmailCategory],
        available_priorities=None,
        available_departments=None,
    ),
    "triage_medium": TaskInfo(
        task_name="triage_medium",
        difficulty="medium",
        instructions=(
            "Triage this email by providing:\n"
            "1. Category (spam, urgent, routine, newsletter, personal)\n"
            "2. Priority level (P0=critical, P1=high, P2=medium, P3=low)\n"
            "3. Target department (engineering, sales, support, hr, legal, marketing, executive)\n"
            "Set category, priority, and department fields in your action."
        ),
        required_fields=["category", "priority", "department"],
        available_categories=[c.value for c in EmailCategory],
        available_priorities=[p.value for p in Priority],
        available_departments=[d.value for d in Department],
    ),
    "full_triage_hard": TaskInfo(
        task_name="full_triage_hard",
        difficulty="hard",
        instructions=(
            "Perform a full email triage:\n"
            "1. Category (spam, urgent, routine, newsletter, personal)\n"
            "2. Priority level (P0=critical, P1=high, P2=medium, P3=low)\n"
            "3. Target department (engineering, sales, support, hr, legal, marketing, executive)\n"
            "4. Draft an appropriate response to this email\n"
            "Set category, priority, department, and response_draft fields in your action. "
            "The response should be professional, address the sender's concerns, "
            "and include next steps where appropriate."
        ),
        required_fields=["category", "priority", "department", "response_draft"],
        available_categories=[c.value for c in EmailCategory],
        available_priorities=[p.value for p in Priority],
        available_departments=[d.value for d in Department],
    ),
}

VALID_TASK_NAMES = list(TASK_DEFINITIONS.keys())


class EmailTriageEnvironment:
    """
    OpenEnv-compliant Email Triage environment.

    Simulates an email inbox where an agent must classify, prioritize,
    route, and optionally draft responses to emails.
    """

    def __init__(self) -> None:
        self._state: Optional[State] = None
        self._ground_truth: Optional[GroundTruth] = None
        self._task_name: Optional[str] = None
        self._email_seed: Optional[int] = None

    # ── OpenEnv interface ────────────────────────────────────────────────

    def reset(
        self,
        task_name: str = "classify_easy",
        seed: int | None = None,
        perturbation: str = "none",
    ) -> Observation:
        """
        Start a new episode.

        Args:
            task_name: One of classify_easy, triage_medium, full_triage_hard
            seed: Optional seed for deterministic email selection
            perturbation: Adversarial perturbation mode. One of: none,
                homoglyph, tone_inversion, identity_spoof, distractor_inject,
                all. Modifies email surface cues while preserving ground truth.

        Returns:
            Initial observation with the email and task instructions.
        """
        if task_name not in TASK_DEFINITIONS:
            return Observation(
                error_message=(
                    f"Unknown task '{task_name}'. "
                    f"Valid tasks: {VALID_TASK_NAMES}"
                ),
                done=True,
            )

        self._task_name = task_name
        self._email_seed = seed
        email, ground_truth = get_random_email(seed=seed, task_name=task_name)
        self._ground_truth = ground_truth

        # Apply adversarial perturbation (surface only — ground truth unchanged)
        if perturbation and perturbation != "none":
            email = apply_perturbation(
                email,
                ground_truth_priority=ground_truth.priority.value,
                mode=perturbation,
                seed=seed,
            )

        self._state = State(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_name=task_name,
            max_steps=1,
            done=False,
        )

        return Observation(
            email=email,
            task=TASK_DEFINITIONS[task_name],
            feedback=None,
            reward=None,
            done=False,
        )

    def step(self, action: TriageAction) -> Observation:
        """
        Execute one step: the agent submits a triage action, receives a grade.

        Args:
            action: The agent's triage decision.

        Returns:
            Observation with reward, feedback, and done=True.
        """
        if self._state is None or self._state.done:
            return Observation(
                error_message="Episode is not active. Call reset() first.",
                done=True,
            )

        if self._ground_truth is None or self._task_name is None:
            return Observation(
                error_message="Internal error: no ground truth available.",
                done=True,
            )

        # Validate required fields
        task_def = TASK_DEFINITIONS[self._task_name]
        missing = []
        for field in task_def.required_fields:
            val = getattr(action, field, None)
            if val is None or (isinstance(val, str) and val.strip() == ""):
                missing.append(field)

        if missing:
            # Penalize but still end the episode.
            # Score must be strictly > 0 per OpenEnv spec.
            self._state.step_count += 1
            self._state.done = True
            return Observation(
                feedback=f"Missing required fields: {missing}. Score: 0.001",
                reward=0.001,
                done=True,
                error_message=f"Required fields not provided: {missing}",
            )

        # Grade the action
        reward, feedback = grade(self._task_name, action, self._ground_truth)

        self._state.step_count += 1
        self._state.done = True

        return Observation(
            feedback=feedback,
            reward=reward,
            done=True,
        )

    def state(self) -> State:
        """Return current episode metadata."""
        if self._state is None:
            return State(
                episode_id="none",
                step_count=0,
                task_name="",
                max_steps=1,
                done=True,
            )
        return self._state.model_copy()
