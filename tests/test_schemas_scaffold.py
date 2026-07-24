"""Tests for shared schema contracts."""

from importlib import import_module

import pytest
from pydantic import ValidationError

from schemas import CameraInfo
from schemas import Frame
from schemas import PipelinePlan
from schemas import SceneGraph
from schemas import SceneObject
from schemas import SceneRelationship
from schemas import SensorCapabilityProfile
from schemas import SensorType
from schemas import SourceDescriptor


def test_schema_objects_validate_required_fields() -> None:
    """Core schema objects should require the fields their callers rely on."""
    source = SourceDescriptor(source_id="cam-1", source_type=SensorType.RGB_CAMERA)
    frame = Frame(frame_id="frame-1", timestamp_ns=42, source_id="cam-1")
    profile = SensorCapabilityProfile(
        source_id="cam-1",
        sensor_type=SensorType.RGB_CAMERA,
        supports_rgb=True,
    )
    plan = PipelinePlan(plan_id="plan-1", capabilities=[profile], selected_models=["model-a"])

    assert source.source_id == "cam-1"
    assert frame.timestamp_ns == 42
    assert profile.supports_rgb is True
    assert plan.capabilities[0].source_id == "cam-1"


@pytest.mark.parametrize(
    ("payload", "expected_message"),
    [
        (
            {"source_id": "cam-1", "sensor_type": "rgb_camera", "unexpected": True},
            "Extra inputs are not permitted",
        ),
        (
            {"source_id": "cam-1"},
            "Field required",
        ),
    ],
)
def test_invalid_capability_payloads_fail_fast(payload: dict[str, object], expected_message: str) -> None:
    """Capability payloads should reject malformed or unexpected input immediately."""
    with pytest.raises(ValidationError, match=expected_message):
        SensorCapabilityProfile.model_validate(payload)


def test_model_and_frame_serialization_roundtrip() -> None:
    """Frame and scene payloads should round-trip through JSON consistently."""
    frame = Frame(
        frame_id="frame-2",
        timestamp_ns=99,
        source_id="cam-2",
        payload={"encoding": "rgb8", "shape": [720, 1280, 3]},
    )
    scene_graph = SceneGraph(
        scene_id="scene-1",
        humans=[SceneObject(object_id="human-1", object_type="person")],
        cameras=[CameraInfo(camera_id="cam-2", width=1280, height=720)],
        sensors=[SourceDescriptor(source_id="cam-2", source_type=SensorType.RGB_CAMERA)],
        relationships=[
            SceneRelationship(
                source_id="human-1",
                target_id="cam-2",
                relation_type="observed_by",
            )
        ],
    )

    frame_roundtrip = Frame.model_validate_json(frame.model_dump_json())
    scene_roundtrip = SceneGraph.model_validate_json(scene_graph.model_dump_json())

    assert frame_roundtrip == frame
    assert scene_roundtrip == scene_graph
    assert scene_roundtrip.sensors[0].source_type is SensorType.RGB_CAMERA


def test_schema_scaffold_public_contract_surface() -> None:
    """The public schema surface should stay explicit and predictable."""
    expected_exports = {
        "BoundingBox",
        "Calibration",
        "CameraInfo",
        "Frame",
        "FusionConfiguration",
        "FusionRequest",
        "FusionResponse",
        "InferenceRequest",
        "InferenceResponse",
        "Mask",
        "Mesh",
        "ModelCapabilities",
        "ModelRequirements",
        "OutputSchema",
        "PipelinePlan",
        "PointCloud",
        "Pose2D",
        "Pose3D",
        "RegistryDependency",
        "RegistryResolution",
        "SceneGraph",
        "SceneObject",
        "SceneRelationship",
        "SensorCapabilityProfile",
        "SensorType",
        "SourceDescriptor",
        "Trajectory",
        "WorkflowEdge",
        "WorkflowGraph",
        "WorkflowNode",
    }

    exported = set(import_module("schemas").__all__)
    assert exported == expected_exports
