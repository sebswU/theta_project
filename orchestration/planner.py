"""Pipeline orchestration contracts.

The orchestration layer maps capabilities to model adapters, fusion plugins,
and execution graphs without embedding model-specific implementation logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from architecture.core_interfaces import PipelinePlanner, WorkflowBuilder
from schemas.capabilities import SensorCapabilityProfile

__all__ = [
    "ExecutionNode",
    "ExecutionGraph",
    "PipelinePlanner",
    "PipelineBuilder",
    "CapabilityMatcher",
    "WorkflowBuilder",
]


@dataclass(slots=True)
class ExecutionNode:
    """Single execution graph node metadata."""

    node_id: str
    node_type: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionGraph:
    """Execution graph description for runtime orchestration."""

    nodes: list[ExecutionNode] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)


class PipelineBuilder(ABC):
    """Builds execution graphs from planner output."""

    @abstractmethod
    def build(self, plan: dict[str, Any]) -> ExecutionGraph:
        """Compile execution graph from plan.

        TODO: Implement execution graph builder.
        """


class CapabilityMatcher(ABC):
    """Matches capabilities against model/plugin requirements."""

    @abstractmethod
    def match(
        self,
        capabilities: list[SensorCapabilityProfile],
        candidate_requirements: dict[str, Any],
    ) -> bool:
        """Check if capabilities satisfy candidate requirements.

        TODO: Implement requirement matching and scoring.
        """
