"""
Typed Pydantic models for the Email Triage OpenEnv environment.

Defines Action, Observation, and State used by both client and server.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────────

class EmailCategory(str, Enum):
    SPAM = "spam"
    URGENT = "urgent"
    ROUTINE = "routine"
    NEWSLETTER = "newsletter"
    PERSONAL = "personal"


class Priority(str, Enum):
    P0 = "P0"  # Critical – immediate action
    P1 = "P1"  # High – same day
    P2 = "P2"  # Medium – within 48h
    P3 = "P3"  # Low – informational


class Department(str, Enum):
    ENGINEERING = "engineering"
    SALES = "sales"
    SUPPORT = "support"
    HR = "hr"
    LEGAL = "legal"
    MARKETING = "marketing"
    EXECUTIVE = "executive"


# ── Action ───────────────────────────────────────────────────────────────────

class TriageAction(BaseModel):
    """Action the agent submits to triage an email."""

    category: str = Field(
        ...,
        description="Email classification: spam, urgent, routine, newsletter, personal",
    )
    priority: Optional[str] = Field(
        None,
        description="Priority level: P0, P1, P2, P3 (required for medium/hard tasks)",
    )
    department: Optional[str] = Field(
        None,
        description="Target department (required for medium/hard tasks)",
    )
    response_draft: Optional[str] = Field(
        None,
        description="Drafted reply to the email (required for hard task)",
    )


# ── Observation ──────────────────────────────────────────────────────────────

class EmailData(BaseModel):
    """An email presented to the agent for triage."""

    sender: str = Field(..., description="Sender email address")
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email body text")
    timestamp: str = Field(..., description="ISO-8601 timestamp")
    has_attachments: bool = Field(False, description="Whether the email has attachments")
    thread_length: int = Field(1, description="Number of messages in thread")


class TaskInfo(BaseModel):
    """Describes what the agent must do."""

    task_name: str = Field(..., description="Task identifier: classify_easy, triage_medium, full_triage_hard")
    difficulty: str = Field(..., description="easy, medium, or hard")
    instructions: str = Field(..., description="Human-readable instructions for the agent")
    required_fields: List[str] = Field(
        ...,
        description="Which TriageAction fields the agent must fill in",
    )
    available_categories: List[str] = Field(
        default_factory=lambda: [c.value for c in EmailCategory],
    )
    available_priorities: Optional[List[str]] = Field(None)
    available_departments: Optional[List[str]] = Field(None)


class Observation(BaseModel):
    """What the agent sees after reset() or step()."""

    email: Optional[EmailData] = Field(None, description="The email to triage")
    task: Optional[TaskInfo] = Field(None, description="Task description and requirements")
    feedback: Optional[str] = Field(None, description="Feedback after the agent submits an action")
    reward: Optional[float] = Field(None, description="Reward for the last action (0.0–1.0)")
    done: bool = Field(False, description="Whether the episode is over")
    error_message: Optional[str] = Field(None, description="Error details if the action was invalid")


# ── State ────────────────────────────────────────────────────────────────────

class State(BaseModel):
    """Episode metadata returned by state()."""

    episode_id: str = Field(..., description="Unique episode identifier")
    step_count: int = Field(0, description="Number of steps taken in this episode")
    task_name: str = Field("", description="Active task name")
    max_steps: int = Field(1, description="Maximum steps allowed for this task")
    done: bool = Field(False, description="Whether the episode has ended")


# ── Ground Truth (internal, not exposed to agent) ────────────────────────────

class GroundTruth(BaseModel):
    """The correct answer for grading. Never sent to the agent."""

    category: EmailCategory
    priority: Priority
    department: Department
    expected_response_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords that should appear in a good response draft",
    )
    expected_response_tone: str = Field(
        "professional",
        description="Expected tone: professional, empathetic, urgent, etc.",
    )
    expected_entities: List[str] = Field(
        default_factory=list,
        description=(
            "Concrete entities (incident IDs, dollar amounts, deadlines, "
            "names) lifted from the email body that a competent response "
            "must reference. Exact-substring matched, case-insensitive."
        ),
    )
    forbidden_phrases: List[str] = Field(
        default_factory=list,
        description=(
            "Generic slop phrases that indicate a lazy, non-specific draft "
            "(e.g. 'thank you for your email'). Each occurrence penalizes "
            "the draft score."
        ),
    )
