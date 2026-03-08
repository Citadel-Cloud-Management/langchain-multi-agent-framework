"""Shared conversation memory for multi-agent workflows."""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_core.messages import BaseMessage


@dataclass
class MemoryEntry:
    """A single entry in the shared memory store."""

    agent_name: str
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
    entry_type: str = "message"


class SharedMemory:
    """Thread-safe shared memory for multi-agent communication.

    Provides a centralized memory store that all agents can read from
    and write to, enabling information sharing across the workflow.
    """

    def __init__(self, max_entries: int = 1000) -> None:
        self._entries: list[MemoryEntry] = []
        self._agent_entries: dict[str, list[MemoryEntry]] = defaultdict(list)
        self._artifacts: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._max_entries = max_entries

    def add_entry(
        self,
        agent_name: str,
        content: str,
        entry_type: str = "message",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Add a memory entry from an agent.

        Args:
            agent_name: Name of the agent adding the entry.
            content: Text content of the entry.
            entry_type: Type classification (message, result, artifact, error).
            metadata: Additional metadata for the entry.
        """
        entry = MemoryEntry(
            agent_name=agent_name,
            content=content,
            entry_type=entry_type,
            metadata=metadata or {},
        )
        with self._lock:
            self._entries.append(entry)
            self._agent_entries[agent_name].append(entry)
            if len(self._entries) > self._max_entries:
                removed = self._entries.pop(0)
                agent_list = self._agent_entries.get(removed.agent_name, [])
                if agent_list and agent_list[0] is removed:
                    agent_list.pop(0)

    def get_recent(self, n: int = 10) -> list[MemoryEntry]:
        """Get the most recent n memory entries.

        Args:
            n: Number of entries to retrieve.

        Returns:
            List of recent MemoryEntry objects.
        """
        with self._lock:
            return list(self._entries[-n:])

    def get_agent_entries(self, agent_name: str, n: int = 10) -> list[MemoryEntry]:
        """Get recent entries from a specific agent.

        Args:
            agent_name: The agent whose entries to retrieve.
            n: Maximum number of entries.

        Returns:
            List of MemoryEntry objects from the specified agent.
        """
        with self._lock:
            return list(self._agent_entries.get(agent_name, [])[-n:])

    def store_artifact(self, key: str, value: Any) -> None:
        """Store a named artifact in shared memory.

        Args:
            key: Unique identifier for the artifact.
            value: The artifact data.
        """
        with self._lock:
            self._artifacts[key] = value

    def get_artifact(self, key: str) -> Optional[Any]:
        """Retrieve a named artifact from shared memory.

        Args:
            key: Identifier of the artifact.

        Returns:
            The artifact data, or None if not found.
        """
        with self._lock:
            return self._artifacts.get(key)

    def get_context_string(self, n: int = 10) -> str:
        """Get a formatted string of recent memory for injection into prompts.

        Args:
            n: Number of recent entries to include.

        Returns:
            Formatted context string.
        """
        entries = self.get_recent(n)
        if not entries:
            return "No previous context available."

        lines = []
        for entry in entries:
            ts = entry.timestamp.strftime("%H:%M:%S")
            lines.append(f"[{ts}] {entry.agent_name} ({entry.entry_type}): {entry.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all memory entries and artifacts."""
        with self._lock:
            self._entries.clear()
            self._agent_entries.clear()
            self._artifacts.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
