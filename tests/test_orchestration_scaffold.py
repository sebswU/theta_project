"""Tests for orchestration planning and workflow graph scaffolding."""

from orchestration import (
    DefaultCapabilityDetector,
    DeterministicPipelinePlanner,
    GraphWorkflowBuilder,
    Orchestrator,
    PlanningConstraints,
)
from schemas import ModelRequirements, SensorCapabilityProfile, SensorType, SourceDescriptor


def _skellycam_sources() -> list[SourceDescriptor]:
    return [
        SourceDescriptor(
            source_id="skellycam-front-rgb",
            source_type=SensorType.RGB_CAMERA,
            metadata={"supports_calibration": True, "supports_multiview": True},
        ),
        SourceDescriptor(
            source_id="skellycam-left-depth",
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


def test_skellycam_capabilities_produce_deterministic_plan() -> None:
    """Equivalent SkellyCam capability inputs should yield stable deterministic plans."""
    detector = DefaultCapabilityDetector()
    planner = _planner()

    caps_a = [detector.detect(source) for source in _skellycam_sources()]
    caps_b = [detector.detect(source) for source in reversed(_skellycam_sources())]

    assert all(isinstance(capability, SensorCapabilityProfile) for capability in caps_a)
    caps_by_source = {capability.source_id: capability for capability in caps_a}
    assert caps_by_source["skellycam-front-rgb"].supports_rgb is True
    assert caps_by_source["skellycam-front-rgb"].supports_calibration is True
    assert caps_by_source["skellycam-front-rgb"].supports_multiview is True
    assert caps_by_source["skellycam-left-depth"].supports_depth is True
    assert caps_by_source["skellycam-left-depth"].supports_synchronization is True

    plan_a = planner.plan(caps_a)
    plan_b = planner.plan(caps_b)

    assert plan_a.plan_id == plan_b.plan_id
    assert plan_a.selected_models == plan_b.selected_models == ["rgb-model", "rgbd-model"]
    assert plan_a.selected_plugins == plan_b.selected_plugins == ["temporal", "volumetric"]


def test_skellycam_planning_respects_allowed_candidates() -> None:
    """SkellyCam planning should strictly respect explicit candidate allow-lists."""
    detector = DefaultCapabilityDetector()
    planner = _planner(
        constraints=PlanningConstraints(
            allowed_models={"rgb-model"},
            allowed_plugins={"temporal"},
        )
    )

    capabilities = [detector.detect(source) for source in _skellycam_sources()]
    plan = planner.plan(capabilities)

    assert plan.selected_models == ["rgb-model"]
    assert plan.selected_plugins == ["temporal"]
    assert set(plan.selected_models).issubset({"rgb-model"})
    assert set(plan.selected_plugins).issubset({"temporal"})


def test_skellycam_workflow_graph_is_structurally_valid() -> None:
    """Workflow graph edges should always reference existing node endpoints."""
    detector = DefaultCapabilityDetector()
    planner = _planner()
    builder = GraphWorkflowBuilder()

    capabilities = [detector.detect(source) for source in _skellycam_sources()]
    plan = planner.plan(capabilities)
    graph = builder.build(plan)

    node_ids = {node.node_id for node in graph.nodes}
    assert graph.nodes
    assert graph.edges
    assert all(edge.source_id in node_ids and edge.target_id in node_ids for edge in graph.edges)


def test_skellycam_workflow_output_matches_plan() -> None:
    """Workflow output should faithfully encode the selected deterministic plan."""
    orchestrator = Orchestrator(
        capability_detector=DefaultCapabilityDetector(),
        planner=_planner(),
        workflow_builder=GraphWorkflowBuilder(),
    )

    plan, graph = orchestrator.create_workflow(_skellycam_sources())

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
