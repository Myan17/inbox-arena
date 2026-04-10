"""
Adversarial Perturbation Engine for InboxArena.

Applies surface-level attacks to emails while preserving ground truth.
This tests whether agents are robust to real-world noise: unicode tricks,
misleading subjects, spoofed senders, and injected distractors.

All perturbations are deterministic given a seed, so scores remain
reproducible even with perturbations enabled.

Perturbation modes:
  - homoglyph:        Replace ASCII chars with visually identical unicode
  - tone_inversion:   Rewrite subject line to contradict body urgency
  - identity_spoof:   Replace sender with a high-authority spoofed name
  - distractor_inject: Insert misleading sentences into the email body
  - all:              Apply all perturbations simultaneously
  - none:             No perturbation (default)
"""

from __future__ import annotations

import random
from typing import Optional

from models import EmailData


# ── Unicode homoglyph map ──────────────────────────────────────────────────
# Characters that look identical to ASCII but are different codepoints.
# Real-world phishing and spam use these to bypass keyword filters.

_HOMOGLYPHS = {
    "a": "\u0430",  # Cyrillic а
    "c": "\u0441",  # Cyrillic с
    "e": "\u0435",  # Cyrillic е
    "o": "\u043e",  # Cyrillic о
    "p": "\u0440",  # Cyrillic р
    "s": "\u0455",  # Cyrillic ѕ
    "x": "\u0445",  # Cyrillic х
    "i": "\u0456",  # Cyrillic і
    "I": "\u0406",  # Cyrillic І
    "S": "\u0405",  # Cyrillic Ѕ
}

# ── Tone inversion templates ───────────────────────────────────────────────
# Subjects that sound casual/low-priority, applied to urgent emails, and
# subjects that sound urgent/high-priority, applied to routine emails.

_CASUAL_SUBJECTS = [
    "Quick question when you get a sec",
    "Hey, small thing",
    "No rush — just wanted to flag this",
    "FYI (non-urgent)",
    "Low-priority housekeeping",
    "Whenever you have a moment...",
]

_PANICKY_SUBJECTS = [
    "🚨🚨🚨 URGENT — NEED RESPONSE NOW 🚨🚨🚨",
    "CRITICAL!!! IMMEDIATE ACTION REQUIRED!!!",
    "⚠️ EMERGENCY — DO NOT IGNORE ⚠️",
    "RED ALERT: RESPOND WITHIN 5 MINUTES",
    "URGENT URGENT URGENT — ESCALATION NOTICE",
]

# ── Identity spoof names ──────────────────────────────────────────────────
# High-authority sender names that might trick an agent into upgrading
# priority based on sender alone.

_SPOOF_SENDERS_HIGH = [
    "ceo@company.com",
    "board-chair@company.com",
    "legal-urgent@company.com",
    "cfo@company.com",
    "vp-engineering@company.com",
]

_SPOOF_SENDERS_LOW = [
    "office-snacks@company.com",
    "intern2026@company.com",
    "team-social@company.com",
    "facilities-request@company.com",
]

# ── Distractor sentences ──────────────────────────────────────────────────
# Misleading sentences injected into the body to test reading comprehension.

_URGENT_DISTRACTORS = [
    "\n\n[Note: The CEO mentioned this is blocking the board meeting tomorrow.]\n",
    "\n\n[FYI: Legal has flagged this as a potential SLA breach.]\n",
    "\n\n[Update: Three customers have escalated about this in the last hour.]\n",
]

_CASUAL_DISTRACTORS = [
    "\n\n[Note: No rush on this — just whenever you get around to it.]\n",
    "\n\n[FYI: This has been sitting in the queue for weeks, no one seems to care.]\n",
    "\n\n[Update: Everyone agrees this is super low priority.]\n",
]


# ── Perturbation functions ────────────────────────────────────────────────

def _apply_homoglyphs(text: str, rng: random.Random, rate: float = 0.15) -> str:
    """Replace ~rate of eligible characters with unicode homoglyphs."""
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch in _HOMOGLYPHS and rng.random() < rate:
            chars[i] = _HOMOGLYPHS[ch]
    return "".join(chars)


def _apply_tone_inversion(
    email: EmailData,
    ground_truth_priority: str,
    rng: random.Random,
) -> EmailData:
    """Replace subject with one that contradicts the actual priority."""
    if ground_truth_priority in ("P0", "P1"):
        # Urgent email gets casual subject
        new_subject = rng.choice(_CASUAL_SUBJECTS)
    else:
        # Low-priority email gets panicky subject
        new_subject = rng.choice(_PANICKY_SUBJECTS)

    return EmailData(
        sender=email.sender,
        subject=new_subject,
        body=email.body,
        timestamp=email.timestamp,
        has_attachments=email.has_attachments,
        thread_length=email.thread_length,
    )


def _apply_identity_spoof(
    email: EmailData,
    ground_truth_priority: str,
    rng: random.Random,
) -> EmailData:
    """Replace sender with a spoofed high/low authority name."""
    if ground_truth_priority in ("P0", "P1"):
        # Urgent email gets a low-authority sender
        new_sender = rng.choice(_SPOOF_SENDERS_LOW)
    else:
        # Low-priority email gets a high-authority sender
        new_sender = rng.choice(_SPOOF_SENDERS_HIGH)

    return EmailData(
        sender=new_sender,
        subject=email.subject,
        body=email.body,
        timestamp=email.timestamp,
        has_attachments=email.has_attachments,
        thread_length=email.thread_length,
    )


def _apply_distractor(
    email: EmailData,
    ground_truth_priority: str,
    rng: random.Random,
) -> EmailData:
    """Inject a misleading sentence into the body."""
    if ground_truth_priority in ("P0", "P1"):
        distractor = rng.choice(_CASUAL_DISTRACTORS)
    else:
        distractor = rng.choice(_URGENT_DISTRACTORS)

    # Insert distractor at a random position in the body
    lines = email.body.split("\n")
    insert_pos = rng.randint(1, max(1, len(lines) - 1))
    lines.insert(insert_pos, distractor)

    return EmailData(
        sender=email.sender,
        subject=email.subject,
        body="\n".join(lines),
        timestamp=email.timestamp,
        has_attachments=email.has_attachments,
        thread_length=email.thread_length,
    )


# ── Public API ─────────────────────────────────────────────────────────────

VALID_PERTURBATIONS = {"none", "homoglyph", "tone_inversion", "identity_spoof", "distractor_inject", "all"}


def apply_perturbation(
    email: EmailData,
    ground_truth_priority: str,
    mode: str = "none",
    seed: Optional[int] = None,
) -> EmailData:
    """
    Apply adversarial perturbations to an email.

    The ground truth is NEVER changed — only the email surface is modified.
    This tests whether agents read the content (body facts, identifiers)
    rather than relying on surface cues (subject tone, sender authority).

    Args:
        email: The original email to perturb.
        ground_truth_priority: The real priority (P0–P3) — used to pick
            the *opposite* surface cue for maximum adversarial effect.
        mode: Perturbation mode (see module docstring).
        seed: Optional seed for deterministic perturbations.

    Returns:
        A new EmailData with perturbations applied (or the original if
        mode is "none").
    """
    if mode == "none" or mode not in VALID_PERTURBATIONS:
        return email

    rng = random.Random(seed)

    if mode == "homoglyph" or mode == "all":
        email = EmailData(
            sender=email.sender,
            subject=_apply_homoglyphs(email.subject, rng),
            body=_apply_homoglyphs(email.body, rng),
            timestamp=email.timestamp,
            has_attachments=email.has_attachments,
            thread_length=email.thread_length,
        )

    if mode == "tone_inversion" or mode == "all":
        email = _apply_tone_inversion(email, ground_truth_priority, rng)

    if mode == "identity_spoof" or mode == "all":
        email = _apply_identity_spoof(email, ground_truth_priority, rng)

    if mode == "distractor_inject" or mode == "all":
        email = _apply_distractor(email, ground_truth_priority, rng)

    return email
