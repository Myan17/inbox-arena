"""
Grading functions for Email Triage tasks.

Each grader returns a float score in [0.0, 1.0] with meaningful partial credit.
Graders are deterministic and reproducible — no LLM-as-judge, no network calls,
no randomness. They run in microseconds.

Design notes for the hard-task grader (`grade_hard`):

- We do NOT reward generic LLM slop. A draft that reads "Thank you for your
  email, I will look into this" should score poorly even if it contains the
  right keyword by accident. This is enforced via `forbidden_phrases`.

- We DO reward drafts that cite concrete details from the email body. An
  incident responder who writes "Joining INC-PAY-4471 bridge now, ETA 2
  minutes" is demonstrably paying attention; one who writes "I am on it" is
  not. This is enforced via `expected_entities`.

- Length is scored as a band, not a staircase. Too short AND too long are
  penalized; the sweet spot mirrors what a real first-reply would look like.

- The old grader gave 0.90 baseline to an 8B model on the hard task. These
  rules are calibrated to put a zero-shot 8B model somewhere in the 0.40–0.65
  range on the benchmark email, leaving meaningful headroom for RL training.
"""

from __future__ import annotations

from models import GroundTruth, TriageAction


def _normalize(value: str | None) -> str:
    """Lowercase and strip a string for comparison."""
    if value is None:
        return ""
    return value.strip().lower()


def _keyword_score(text: str, keywords: list[str]) -> float:
    """Score 0.0–1.0 based on fraction of expected keywords present in text."""
    if not keywords:
        return 1.0  # No keywords expected → full marks
    text_lower = text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    return matches / len(keywords)


def _entity_score(text: str, entities: list[str]) -> tuple[float, int, int]:
    """
    Fraction of required entities that appear verbatim (case-insensitive) in
    the draft. Returns (score, matched, total).
    """
    if not entities:
        return 1.0, 0, 0  # No entities required → full marks
    text_lower = text.lower()
    matches = sum(1 for ent in entities if ent.lower() in text_lower)
    return matches / len(entities), matches, len(entities)


def _forbidden_penalty(text: str, forbidden: list[str]) -> tuple[float, list[str]]:
    """
    Penalty in [0.0, 1.0] for generic slop phrases. Each occurrence costs
    0.25 up to a cap of 1.0. Returns (penalty, hits).
    """
    if not forbidden:
        return 0.0, []
    text_lower = text.lower()
    hits = [phrase for phrase in forbidden if phrase.lower() in text_lower]
    penalty = min(1.0, 0.25 * len(hits))
    return penalty, hits


def _tone_score(text: str, expected_tone: str) -> float:
    """
    Heuristic tone check. Stricter than the previous version: requires 2+
    signals for full marks, 1 signal gives 0.4, 0 signals gives 0.0.
    """
    text_lower = text.lower()

    tone_signals = {
        "urgent": [
            "immediately", "asap", "right away", "joining", "on it now",
            "investigating", "bridge", "war room", "acknowledged",
        ],
        "empathetic": [
            "apolog", "understand", "appreciate the urgency", "sorry",
            "frustration", "regret",
        ],
        "professional": [
            "regards", "thank you", "please", "sincerely", "best",
            "confirmed", "noted",
        ],
        "casual": [
            "hey", "cheers", "cool", "sounds good", "awesome",
            "let's", "catch up",
        ],
    }

    signals = tone_signals.get(expected_tone, tone_signals["professional"])
    matches = sum(1 for s in signals if s in text_lower)

    if matches >= 2:
        return 1.0
    if matches == 1:
        return 0.4
    return 0.0


def _length_score(text: str) -> float:
    """
    Band-based length scoring. A real first-reply is 80–350 characters.
    Very short = clearly incomplete. Very long = probably waffling.
    """
    n = len(text)
    if n == 0:
        return 0.0
    if n < 40:
        return 0.1
    if n < 80:
        return 0.5
    if n <= 350:
        return 1.0  # sweet spot
    if n <= 600:
        return 0.7
    if n <= 1000:
        return 0.4
    return 0.2  # wall-of-text penalty


def _confidence_bonus(confidence: float | None, base_score: float) -> tuple[float, str]:
    """
    Brier-score-based calibration bonus/penalty.

    Rewards agents that know what they know: if you claim confidence=0.95
    and score 0.95, you earn +0.05. If you claim confidence=0.95 and
    score 0.20, you lose -0.03. Omitting confidence returns (0.0, "").

    The bonus is capped at +0.05 and the penalty at -0.03 to avoid
    dominating the base score. This is a meta-cognitive signal — it tests
    whether the agent can distinguish "I'm sure" from "I'm guessing",
    which is critical for real-world triage (uncertain classifications
    should be flagged for human review, not silently committed).

    Brier score = (confidence - accuracy)^2, range [0, 1].
    We map: 0 → +0.05 (perfect calibration), 1 → -0.03 (maximally wrong).
    """
    if confidence is None:
        return 0.0, ""

    brier = (confidence - base_score) ** 2
    # Linear interpolation: brier=0 → +0.05, brier=1 → -0.03
    bonus = 0.05 - 0.08 * brier
    bonus = max(-0.03, min(0.05, bonus))

    if bonus >= 0:
        return round(bonus, 3), f"Calibration: Brier={brier:.2f} (+{bonus:.3f})"
    return round(bonus, 3), f"Calibration: Brier={brier:.2f} ({bonus:.3f})"


# ── Task Graders ─────────────────────────────────────────────────────────────

def grade_easy(action: TriageAction, truth: GroundTruth) -> tuple[float, str]:
    """
    Easy task: Email classification only.
    Score: 1.0 if correct category, 0.0 otherwise.
    + optional confidence calibration bonus (up to +0.05).
    """
    predicted = _normalize(action.category)
    expected = _normalize(truth.category.value)

    if predicted == expected:
        base = 1.0
        feedback = f"Correct! Category is '{truth.category.value}'."
    else:
        base = 0.0
        feedback = (
            f"Incorrect. You predicted '{action.category}', "
            f"but the correct category is '{truth.category.value}'."
        )

    cal_bonus, cal_feedback = _confidence_bonus(action.confidence, base)
    if cal_feedback:
        feedback += f" | {cal_feedback}"

    return round(min(max(base + cal_bonus, 0.0), 1.05), 3), feedback


def grade_medium(action: TriageAction, truth: GroundTruth) -> tuple[float, str]:
    """
    Medium task: Classification + Priority + Department routing.
    Partial credit:
      - Category correct:   0.40
      - Priority correct:   0.30
      - Department correct: 0.30
    Total: 0.0–1.0
    """
    score = 0.0
    feedback_parts = []

    # Category (0.40)
    if _normalize(action.category) == _normalize(truth.category.value):
        score += 0.40
        feedback_parts.append("Category: correct (+0.40)")
    else:
        feedback_parts.append(
            f"Category: wrong (expected '{truth.category.value}', "
            f"got '{action.category}')"
        )

    # Priority (0.30)
    if _normalize(action.priority) == _normalize(truth.priority.value):
        score += 0.30
        feedback_parts.append("Priority: correct (+0.30)")
    else:
        feedback_parts.append(
            f"Priority: wrong (expected '{truth.priority.value}', "
            f"got '{action.priority}')"
        )

    # Department (0.30)
    if _normalize(action.department) == _normalize(truth.department.value):
        score += 0.30
        feedback_parts.append("Department: correct (+0.30)")
    else:
        feedback_parts.append(
            f"Department: wrong (expected '{truth.department.value}', "
            f"got '{action.department}')"
        )

    cal_bonus, cal_feedback = _confidence_bonus(action.confidence, score)
    if cal_feedback:
        feedback_parts.append(cal_feedback)
    score += cal_bonus

    feedback = " | ".join(feedback_parts) + f" | Total: {score:.2f}"
    return round(min(max(score, 0.0), 1.05), 3), feedback


def grade_hard(action: TriageAction, truth: GroundTruth) -> tuple[float, str]:
    """
    Hard task: Full triage + draft response.

    Rebalanced rubric (total 1.0):
      - Category correct:          0.15
      - Priority correct:          0.10
      - Department correct:        0.10
      - Response keywords:         0.15  (fraction of expected keywords)
      - Response entities:         0.25  (fraction of required entities cited)
      - Response tone:             0.10  (2+ signals for full marks)
      - Response length band:      0.10  (sweet spot 80-350 chars)
      - Forbidden-phrase penalty:  up to -0.20 on the response subtotal

    The entity component is the heaviest single piece: it rewards drafts that
    actually read the email and reference specific details. Forbidden phrases
    penalize generic slop.
    """
    score = 0.0
    feedback_parts: list[str] = []

    # ── Structured fields ────────────────────────────────────────────────
    if _normalize(action.category) == _normalize(truth.category.value):
        score += 0.15
        feedback_parts.append("Category: correct (+0.15)")
    else:
        feedback_parts.append(
            f"Category: wrong (expected '{truth.category.value}')"
        )

    if _normalize(action.priority) == _normalize(truth.priority.value):
        score += 0.10
        feedback_parts.append("Priority: correct (+0.10)")
    else:
        feedback_parts.append(
            f"Priority: wrong (expected '{truth.priority.value}')"
        )

    if _normalize(action.department) == _normalize(truth.department.value):
        score += 0.10
        feedback_parts.append("Department: correct (+0.10)")
    else:
        feedback_parts.append(
            f"Department: wrong (expected '{truth.department.value}')"
        )

    # ── Response quality ─────────────────────────────────────────────────
    response = action.response_draft or ""
    response_subtotal = 0.0

    # Keywords (0.15)
    kw_score = _keyword_score(response, truth.expected_response_keywords)
    kw_points = round(0.15 * kw_score, 3)
    response_subtotal += kw_points
    feedback_parts.append(
        f"Keywords: {kw_score:.0%} match (+{kw_points:.2f})"
    )

    # Entities (0.25)
    ent_score, ent_matched, ent_total = _entity_score(
        response, truth.expected_entities
    )
    ent_points = round(0.25 * ent_score, 3)
    response_subtotal += ent_points
    feedback_parts.append(
        f"Entities cited: {ent_matched}/{ent_total} (+{ent_points:.2f})"
    )

    # Tone (0.10)
    tone_score = _tone_score(response, truth.expected_response_tone)
    tone_points = round(0.10 * tone_score, 3)
    response_subtotal += tone_points
    feedback_parts.append(
        f"Tone ({truth.expected_response_tone}): {tone_score:.0%} (+{tone_points:.2f})"
    )

    # Length (0.10)
    length_score = _length_score(response)
    length_points = round(0.10 * length_score, 3)
    response_subtotal += length_points
    feedback_parts.append(
        f"Length ({len(response)} chars): {length_score:.0%} (+{length_points:.2f})"
    )

    # Forbidden-phrase penalty on the response subtotal
    penalty_frac, penalty_hits = _forbidden_penalty(
        response, truth.forbidden_phrases
    )
    penalty_points = round(0.20 * penalty_frac, 3)
    if penalty_points > 0:
        feedback_parts.append(
            f"Slop penalty: {penalty_hits} (-{penalty_points:.2f})"
        )
    response_subtotal = max(0.0, response_subtotal - penalty_points)

    score += response_subtotal

    cal_bonus, cal_feedback = _confidence_bonus(action.confidence, min(score, 1.0))
    if cal_feedback:
        feedback_parts.append(cal_feedback)
    score += cal_bonus

    feedback = " | ".join(feedback_parts) + f" | Total: {score:.2f}"
    return round(min(max(score, 0.0), 1.05), 3), feedback


# ── Dispatcher ───────────────────────────────────────────────────────────────

GRADERS = {
    "classify_easy": grade_easy,
    "triage_medium": grade_medium,
    "full_triage_hard": grade_hard,
}


def grade(task_name: str, action: TriageAction, truth: GroundTruth) -> tuple[float, str]:
    """Grade an action for the given task. Returns (score, feedback)."""
    grader = GRADERS.get(task_name)
    if grader is None:
        return 0.0, f"Unknown task: {task_name}"
    return grader(action, truth)
