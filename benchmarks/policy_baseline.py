"""Score a ladder of scripted policies against the live graders.

Why this exists: the README quotes a single LLM baseline, which needs an API
key and a hosted model to reproduce and drifts whenever the grader is retuned.
This benchmark instead scores six fixed policies -- from degenerate to oracle --
with no model, no key and no network. It answers a question the LLM number
cannot: *does the reward function actually discriminate?* A grader that scores
a constant guesser near an attentive agent is not a usable training signal.

Run: python benchmarks/policy_baseline.py [--episodes 60] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))

from models import GroundTruth, TriageAction  # noqa: E402
from server.data import get_random_email  # noqa: E402
from server.graders import grade  # noqa: E402

TASKS = ["classify_easy", "triage_medium", "full_triage_hard"]

CATEGORIES = ["spam", "urgent", "routine", "newsletter", "personal"]
PRIORITIES = ["P0", "P1", "P2", "P3"]
DEPARTMENTS = ["engineering", "sales", "support", "hr", "legal", "marketing", "executive"]

SLOP = (
    "Thank you for reaching out. I will look into this and get back to you as "
    "soon as possible. Please let me know if you have any further questions."
)


# ── Policies ────────────────────────────────────────────────────────────────
# Each takes (email, truth, rng) and returns a TriageAction. `truth` is passed
# so the upper rungs can construct a deliberately good answer; the lower rungs
# ignore it entirely.

def policy_constant(email, truth, rng) -> TriageAction:
    """Always guesses the same thing. The floor any real agent must clear."""
    return TriageAction(category="routine", priority="P2", department="support",
                        response_draft=SLOP)


def policy_random(email, truth, rng) -> TriageAction:
    """Uniform random over the label space."""
    return TriageAction(
        category=rng.choice(CATEGORIES),
        priority=rng.choice(PRIORITIES),
        department=rng.choice(DEPARTMENTS),
        response_draft=SLOP,
    )


def policy_keyword(email, truth, rng) -> TriageAction:
    """A plausible non-ML heuristic over the subject and body."""
    text = f"{email.subject} {email.body}".lower()
    if any(w in text for w in ("unsubscribe", "limited offer", "winner", "click here")):
        category, priority, dept = "spam", "P3", "support"
    elif any(w in text for w in ("outage", "down", "urgent", "asap", "incident", "sev")):
        category, priority, dept = "urgent", "P0", "engineering"
    elif "newsletter" in text or "digest" in text:
        category, priority, dept = "newsletter", "P3", "marketing"
    else:
        category, priority, dept = "routine", "P2", "support"
    return TriageAction(category=category, priority=priority, department=dept,
                        response_draft=SLOP)


def policy_fields_only(email, truth, rng) -> TriageAction:
    """Perfect structured fields, no draft. Isolates the response weighting."""
    return TriageAction(
        category=truth.category.value,
        priority=truth.priority.value,
        department=truth.department.value,
        response_draft="",
    )


def policy_slop(email, truth, rng) -> TriageAction:
    """Perfect fields plus a fluent but contentless draft."""
    return TriageAction(
        category=truth.category.value,
        priority=truth.priority.value,
        department=truth.department.value,
        response_draft=SLOP,
    )


def policy_attentive(email, truth, rng) -> TriageAction:
    """Perfect fields plus a draft citing the required entities and keywords."""
    entities = " ".join(truth.expected_entities)
    keywords = " ".join(truth.expected_response_keywords)
    draft = (
        f"Acknowledged {entities}. Acting on this now -- {keywords}. "
        f"I have the details from your message and will follow up with status "
        f"as soon as the next checkpoint lands."
    ).strip()
    return TriageAction(
        category=truth.category.value,
        priority=truth.priority.value,
        department=truth.department.value,
        response_draft=draft,
        confidence=0.9,
    )


POLICIES = [
    ("constant", policy_constant),
    ("random", policy_random),
    ("keyword-heuristic", policy_keyword),
    ("fields-only (no draft)", policy_fields_only),
    ("perfect fields + slop draft", policy_slop),
    ("perfect fields + attentive draft", policy_attentive),
]


def evaluate(policy_fn, task: str, episodes: int) -> list[float]:
    rng = random.Random(0)
    scores = []
    for seed in range(episodes):
        email, truth = get_random_email(seed=seed, task_name=task)
        action = policy_fn(email, truth, rng)
        score, _ = grade(task, action, truth)
        scores.append(score)
    return scores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    print(f"InboxArena policy baseline -- {args.episodes} episodes per task, "
          f"seeds 0..{args.episodes - 1}, no model required\n")

    header = f"{'policy':<34}" + "".join(f"{t:>20}" for t in TASKS)
    print(header)
    print("-" * len(header))

    rows = []
    for name, fn in POLICIES:
        means = {}
        line = f"{name:<34}"
        for task in TASKS:
            scores = evaluate(fn, task, args.episodes)
            means[task] = statistics.mean(scores)
            line += f"{means[task]:>20.3f}"
        print(line)
        rows.append({"policy": name, **{t: round(means[t], 4) for t in TASKS}})

    print()
    floor = rows[0]
    ceiling = rows[-1]
    for task in TASKS:
        spread = ceiling[task] - floor[task]
        print(f"{task:<20} discrimination (attentive - constant): {spread:+.3f}")

    if args.json:
        args.json.write_text(json.dumps({"episodes": args.episodes, "results": rows}, indent=2) + "\n")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
