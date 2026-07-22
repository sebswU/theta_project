"""Pipeline orchestration contracts.

The orchestration layer maps capabilities to model adapters, fusion plugins,
and execution graphs without embedding model-specific implementation logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from schemas.capabilities import SensorCapabilityProfile


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


class PipelinePlanner(ABC):
    """Selects models/fusion/output pipeline from capabilities."""

    @abstractmethod
    def plan(self, capabilities: list[SensorCapabilityProfile]) -> dict[str, Any]:
        """Create a model and fusion selection plan.

        TODO: Implement planning policy.
        """


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


class WorkflowBuilder(ABC):
    """Creates deployable workflow artifacts from an execution graph."""

    @abstractmethod
    def build_workflow(self, graph: ExecutionGraph) -> dict[str, Any]:
        """Build workflow descriptor for execution backends.

        TODO: Implement backend-specific workflow generation.
        """
