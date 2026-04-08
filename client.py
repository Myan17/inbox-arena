"""
HTTP client for the Email Triage OpenEnv environment.

Provides a typed Python interface for interacting with the environment server.
"""

from __future__ import annotations

from typing import Optional

import httpx

from models import Observation, State, TriageAction


class EmailTriageClient:
    """
    Client for the Email Triage environment HTTP API.

    Usage:
        client = EmailTriageClient("http://localhost:7860")
        obs, state = client.reset(task_name="classify_easy", seed=42)
        result = client.step(TriageAction(category="spam"))
        print(result.observation.reward)
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def reset(
        self,
        task_name: str = "classify_easy",
        seed: Optional[int] = None,
    ) -> tuple[Observation, State]:
        """Reset the environment and start a new episode."""
        payload = {"task_name": task_name}
        if seed is not None:
            payload["seed"] = seed

        response = self._client.post("/reset", json=payload)
        response.raise_for_status()
        data = response.json()

        observation = Observation.model_validate(data["observation"])
        state = State.model_validate(data["state"])
        return observation, state

    def step(self, action: TriageAction) -> tuple[Observation, float | None, bool, State]:
        """Submit a triage action and get the graded result."""
        response = self._client.post(
            "/step",
            json={"action": action.model_dump()},
        )
        response.raise_for_status()
        data = response.json()

        observation = Observation.model_validate(data["observation"])
        reward = data.get("reward")
        done = data.get("done", False)
        state = State.model_validate(data["state"])
        return observation, reward, done, state

    def state(self) -> State:
        """Get current episode metadata."""
        response = self._client.get("/state")
        response.raise_for_status()
        return State.model_validate(response.json())

    def tasks(self) -> list[str]:
        """List available tasks."""
        response = self._client.get("/tasks")
        response.raise_for_status()
        return response.json()["tasks"]

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
