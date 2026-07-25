"""Tests for orchestration planning and workflow graph scaffolding."""

from orchestration import (
    DefaultCapabilityDetector,
    DeterministicPipelinePlanner,
    GraphWorkflowBuilder,
    Orchestrator,
    PlanningConstraints,
)
from schemas import ModelRequirements, SensorType, SourceDescriptor


def _sources() -> list[SourceDescriptor]:
    return [
        SourceDescriptor(
            source_id="cam-rgb",
            source_type=SensorType.RGB_CAMERA,
            metadata={"supports_calibration": True, "supports_multiview": True},
        ),
        SourceDescriptor(
            source_id="cam-depth",
            source_type=SensorType.DEPTH_CAMERA,
            metadata={"supports_synchronization": True},
        ),
    ]


def _planner(constraints: PlanningConstraints | None = None) -> DeterministicPipelinePlanner:
    return DeterministicPipelinePlanner(
        model_requirements={
            "rgb-model": ModelRequirements(
                model_name="rgb-model",
                required_sources=[SensorType.RGB_CAMERA],
            ),
            "thermal-model": ModelRequirements(
                model_name="thermal-model",
                required_sources=[SensorType.THERMAL_CAMERA],
            ),
            "rgbd-model": ModelRequirements(
                model_name="rgbd-model",
                required_sources=[SensorType.RGB_CAMERA, SensorType.DEPTH_CAMERA],
            ),
        },
        plugin_requirements={
            "temporal": {"required_sources": [SensorType.RGB_CAMERA]},
            "volumetric": {"required_sources": [SensorType.DEPTH_CAMERA]},
            "thermal-only": {"required_sources": [SensorType.THERMAL_CAMERA]},
        },
        constraints=constraints,
    )


def test_capabilities_produce_deterministic_plans() -> None:
    """Equivalent source capability inputs should yield stable plan IDs and choices."""
    detector = DefaultCapabilityDetector()
    planner = _planner()

    caps_a = [detector.detect(source) for source in _sources()]
    caps_b = [detector.detect(source) for source in reversed(_sources())]

    plan_a = planner.plan(caps_a)
    plan_b = planner.plan(caps_b)

    assert plan_a.plan_id == plan_b.plan_id
    assert plan_a.selected_models == plan_b.selected_models == ["rgb-model", "rgbd-model"]
    assert plan_a.selected_plugins == plan_b.selected_plugins == ["temporal", "volumetric"]


def test_planning_respects_candidate_constraints() -> None:
    """Planner should only select candidates allowed by explicit constraints."""
    detector = DefaultCapabilityDetector()
    planner = _planner(
        constraints=PlanningConstraints(
            allowed_models={"rgb-model"},
            allowed_plugins={"temporal"},
        )
    )

    capabilities = [detector.detect(source) for source in _sources()]
    plan = planner.plan(capabilities)

    assert plan.selected_models == ["rgb-model"]
    assert plan.selected_plugins == ["temporal"]


def test_execution_graph_is_structurally_valid() -> None:
    """Workflow graph should have valid node endpoints for every edge."""
    detector = DefaultCapabilityDetector()
    planner = _planner()
    builder = GraphWorkflowBuilder()

    capabilities = [detector.detect(source) for source in _sources()]
    plan = planner.plan(capabilities)
    graph = builder.build(plan)

    node_ids = {node.node_id for node in graph.nodes}
    assert graph.nodes
    assert graph.edges
    assert all(edge.source_id in node_ids and edge.target_id in node_ids for edge in graph.edges)


def test_workflow_output_matches_graph() -> None:
    """Orchestrator workflow output should encode selected plan components."""
    orchestrator = Orchestrator(
        capability_detector=DefaultCapabilityDetector(),
        planner=_planner(),
        workflow_builder=GraphWorkflowBuilder(),
    )

    plan, graph = orchestrator.create_workflow(_sources())

    graph_model_names = sorted(
        node.config["name"]
        for node in graph.nodes
        if node.node_type == "model"
    )
    graph_plugin_names = sorted(
        node.config["name"]
        for node in graph.nodes
        if node.node_type == "fusion_plugin"
    )

    assert graph.workflow_id.endswith(plan.plan_id)
    assert graph.metadata["plan_id"] == plan.plan_id
    assert graph_model_names == sorted(plan.selected_models)
    assert graph_plugin_names == sorted(plan.selected_plugins)


def test_orchestration_scaffold_suite_runs() -> None:
    """Sentinel to keep explicit orchestration scaffold suite presence."""
    assert True
