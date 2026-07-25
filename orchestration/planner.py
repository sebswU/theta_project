"""Pipeline orchestration contracts.

The orchestration layer maps capabilities to model adapters, fusion plugins,
and execution graphs without embedding model-specific implementation logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from acquisition.capability_detection import CapabilityDetector as CapabilityDetectorContract
from architecture.core_interfaces import PipelinePlanner, WorkflowBuilder
from schemas.capabilities import SensorCapabilityProfile, SensorType
from schemas.models import (
    ModelRequirements,
    PipelinePlan,
    SourceDescriptor,
    WorkflowEdge,
    WorkflowGraph,
    WorkflowNode,
)

__all__ = [
    "ExecutionNode",
    "ExecutionGraph",
    "PipelinePlanner",
    "PipelineBuilder",
    "CapabilityMatcher",
    "DefaultCapabilityDetector",
    "DeterministicPipelinePlanner",
    "GraphWorkflowBuilder",
    "Orchestrator",
    "PlanningConstraints",
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


class PipelineBuilder(WorkflowBuilder):
    """Compatibility alias for workflow graph builders."""


@dataclass(slots=True)
class PlanningConstraints:
    """Deterministic candidate constraints for planner selection."""

    allowed_models: set[str] | None = None
    allowed_plugins: set[str] | None = None


class CapabilityMatcher:
    """Matches capabilities against model/plugin requirements."""

    def match(
        self,
        capabilities: list[SensorCapabilityProfile],
        candidate_requirements: ModelRequirements,
    ) -> bool:
        """Check if capabilities satisfy model requirements deterministically."""
        available_types = {cap.sensor_type for cap in capabilities}
        required = set(candidate_requirements.required_sources)
        return required.issubset(available_types)


class DefaultCapabilityDetector(CapabilityDetectorContract):
    """Deterministic capability detector from source descriptors."""

    def detect(self, source_descriptor: SourceDescriptor) -> SensorCapabilityProfile:
        """Map source descriptor type to normalized planner-facing capabilities."""
        sensor_type = source_descriptor.source_type
        return SensorCapabilityProfile(
            source_id=source_descriptor.source_id,
            sensor_type=sensor_type,
            supports_rgb=sensor_type in {SensorType.RGB_CAMERA, SensorType.DATASET},
            supports_depth=sensor_type in {SensorType.DEPTH_CAMERA, SensorType.LIDAR},
            supports_thermal=sensor_type is SensorType.THERMAL_CAMERA,
            supports_imu=sensor_type is SensorType.IMU,
            supports_synchronization=bool(
                source_descriptor.metadata.get("supports_synchronization", False)
            ),
            supports_calibration=bool(
                source_descriptor.metadata.get("supports_calibration", False)
            ),
            supports_multiview=bool(source_descriptor.metadata.get("supports_multiview", False)),
        )


class DeterministicPipelinePlanner(PipelinePlanner):
    """Planner that deterministically selects models/plugins under constraints."""

    def __init__(
        self,
        model_requirements: dict[str, ModelRequirements],
        plugin_requirements: dict[str, dict[str, Any]] | None = None,
        constraints: PlanningConstraints | None = None,
        matcher: CapabilityMatcher | None = None,
    ) -> None:
        self._model_requirements = dict(model_requirements)
        self._plugin_requirements = dict(plugin_requirements or {})
        self._constraints = constraints or PlanningConstraints()
        self._matcher = matcher or CapabilityMatcher()

    def plan(self, capabilities: list[SensorCapabilityProfile]) -> PipelinePlan:
        ordered_caps = sorted(capabilities, key=lambda item: item.source_id)
        selected_models = self._select_models(ordered_caps)
        selected_plugins = self._select_plugins(ordered_caps)
        plan_id = self._build_plan_id(ordered_caps, selected_models, selected_plugins)

        return PipelinePlan(
            plan_id=plan_id,
            capabilities=ordered_caps,
            selected_models=selected_models,
            selected_plugins=selected_plugins,
            parameters={
                "capability_count": len(ordered_caps),
                "selected_model_count": len(selected_models),
                "selected_plugin_count": len(selected_plugins),
            },
        )

    def _select_models(self, capabilities: list[SensorCapabilityProfile]) -> list[str]:
        selected: list[str] = []
        for model_name in sorted(self._model_requirements):
            if (
                self._constraints.allowed_models is not None
                and model_name not in self._constraints.allowed_models
            ):
                continue
            requirements = self._model_requirements[model_name]
            if self._matcher.match(capabilities, requirements):
                selected.append(model_name)
        return selected

    def _select_plugins(self, capabilities: list[SensorCapabilityProfile]) -> list[str]:
        selected: list[str] = []
        available_types = {cap.sensor_type for cap in capabilities}

        for plugin_name in sorted(self._plugin_requirements):
            if (
                self._constraints.allowed_plugins is not None
                and plugin_name not in self._constraints.allowed_plugins
            ):
                continue

            required_sources = self._plugin_requirements[plugin_name].get("required_sources", [])
            required_set = {
                source if isinstance(source, SensorType) else SensorType(source)
                for source in required_sources
            }
            if required_set.issubset(available_types):
                selected.append(plugin_name)
        return selected

    @staticmethod
    def _build_plan_id(
        capabilities: list[SensorCapabilityProfile],
        selected_models: list[str],
        selected_plugins: list[str],
    ) -> str:
        sources = "+".join(cap.source_id for cap in capabilities) or "none"
        models = "+".join(selected_models) or "none"
        plugins = "+".join(selected_plugins) or "none"
        return f"plan::{sources}::{models}::{plugins}"


class GraphWorkflowBuilder(WorkflowBuilder):
    """Builds deterministic workflow graphs from a pipeline plan."""

    def build(self, plan: PipelinePlan) -> WorkflowGraph:
        source_nodes = [
            WorkflowNode(node_id=f"source:{cap.source_id}", node_type="source")
            for cap in plan.capabilities
        ]
        model_nodes = [
            WorkflowNode(node_id=f"model:{name}", node_type="model", config={"name": name})
            for name in plan.selected_models
        ]
        plugin_nodes = [
            WorkflowNode(node_id=f"plugin:{name}", node_type="fusion_plugin", config={"name": name})
            for name in plan.selected_plugins
        ]
        output_node = WorkflowNode(node_id="output:workflow", node_type="workflow_output")

        edges: list[WorkflowEdge] = []
        if model_nodes:
            for source in source_nodes:
                for model in model_nodes:
                    edges.append(
                        WorkflowEdge(
                            source_id=source.node_id,
                            target_id=model.node_id,
                            relation_type="feeds",
                        )
                    )

        upstream_nodes = model_nodes or source_nodes
        if plugin_nodes:
            for upstream in upstream_nodes:
                for plugin in plugin_nodes:
                    edges.append(
                        WorkflowEdge(
                            source_id=upstream.node_id,
                            target_id=plugin.node_id,
                            relation_type="feeds",
                        )
                    )
            for plugin in plugin_nodes:
                edges.append(
                    WorkflowEdge(
                        source_id=plugin.node_id,
                        target_id=output_node.node_id,
                        relation_type="produces",
                    )
                )
        else:
            for upstream in upstream_nodes:
                edges.append(
                    WorkflowEdge(
                        source_id=upstream.node_id,
                        target_id=output_node.node_id,
                        relation_type="produces",
                    )
                )

        nodes = source_nodes + model_nodes + plugin_nodes + [output_node]
        node_ids = {node.node_id for node in nodes}
        for edge in edges:
            if edge.source_id not in node_ids or edge.target_id not in node_ids:
                raise ValueError("Workflow graph contains edge with missing endpoint node")

        return WorkflowGraph(
            workflow_id=f"workflow::{plan.plan_id}",
            nodes=nodes,
            edges=edges,
            metadata={
                "plan_id": plan.plan_id,
                "selected_models": list(plan.selected_models),
                "selected_plugins": list(plan.selected_plugins),
            },
        )


class Orchestrator:
    """Wires capability detection, planning, and graph building."""

    def __init__(
        self,
        capability_detector: CapabilityDetectorContract,
        planner: PipelinePlanner,
        workflow_builder: WorkflowBuilder,
    ) -> None:
        self._capability_detector = capability_detector
        self._planner = planner
        self._workflow_builder = workflow_builder

    def create_workflow(
        self, sources: list[SourceDescriptor]
    ) -> tuple[PipelinePlan, WorkflowGraph]:
        """Create plan and workflow graph from source descriptors."""
        capabilities = [
            self._capability_detector.detect(source)
            for source in sorted(sources, key=lambda item: item.source_id)
        ]
        plan = self._planner.plan(capabilities)
        graph = self._workflow_builder.build(plan)
        return plan, graph
