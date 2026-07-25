"""Tests for registry-facing CVPR model adapter scaffolding."""

import pytest

from registry import CVPRModel, CVPRModelLifecycleError, ModelRegistry
from schemas import (
    Frame,
    InferenceRequest,
    InferenceResponse,
    ModelCapabilities,
    ModelRequirements,
    OutputSchema,
    SensorType,
)


class _DemoModel(CVPRModel):
    """Test adapter implementing the canonical CVPR model contract."""

    @property
    def model_name(self) -> str:
        return "demo-model"

    def _load(self) -> None:
        return

    def _infer(self, inputs: InferenceRequest) -> InferenceResponse:
        return InferenceResponse(
            request_id=inputs.request_id,
            outputs={"frame_count": len(inputs.frames)},
        )

    def _validate_inputs(self, inputs: InferenceRequest) -> bool:
        return all(frame.source_id.startswith("cam-") for frame in inputs.frames)

    def _get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_name=self.model_name,
            supported_sources=[SensorType.RGB_CAMERA],
            output_types=["detections"],
            metadata={"task": "detection"},
        )

    def _get_requirements(self) -> ModelRequirements:
        return ModelRequirements(
            model_name=self.model_name,
            required_sources=[SensorType.RGB_CAMERA],
            minimum_version="1.0.0",
            resources={"dependencies": ["numpy>=1.26", {"package_name": "torch"}]},
            metadata={"device": "cpu"},
        )

    def _output_schema(self) -> OutputSchema:
        return OutputSchema(name="detections", fields=["boxes", "scores", "labels"])


def _valid_request() -> InferenceRequest:
    return InferenceRequest(
        request_id="req-1",
        frames=[Frame(frame_id="f-1", timestamp_ns=1, source_id="cam-1")],
    )


def test_required_cvpr_methods_exist() -> None:
    """Base adapter should expose required public methods."""
    model = _DemoModel()

    assert callable(model.load)
    assert callable(model.infer)
    assert callable(model.validate_inputs)
    assert callable(model.get_capabilities)
    assert callable(model.get_requirements)
    assert callable(model.output_schema)


def test_capability_and_requirement_metadata_are_consistent() -> None:
    """Capabilities and requirements should expose matching model identity."""
    model = _DemoModel()

    capabilities = model.get_capabilities()
    requirements = model.get_requirements()

    assert capabilities.model_name == model.model_name
    assert requirements.model_name == model.model_name
    assert capabilities.metadata["task"] == "detection"
    assert requirements.metadata["device"] == "cpu"


def test_invalid_inference_inputs_are_rejected() -> None:
    """Inference should reject bad lifecycle order and invalid request content."""
    model = _DemoModel()

    with pytest.raises(CVPRModelLifecycleError, match="before load"):
        model.infer(_valid_request())

    model.load()

    with pytest.raises(ValueError, match="Invalid inference request"):
        model.infer(InferenceRequest(request_id="req-2", frames=[]))

    with pytest.raises(ValueError, match="Invalid inference request"):
        model.infer(
            InferenceRequest(
                request_id="req-3",
                frames=[Frame(frame_id="f-2", timestamp_ns=2, source_id="sensor-1")],
            )
        )


def test_model_registration_and_lookup_work() -> None:
    """Registry should register, discover, validate, instantiate, and resolve deps."""
    registry = ModelRegistry()

    registry.register("demo", _DemoModel)

    assert registry.discover() == ["demo"]
    assert registry.validate("demo") is True
    assert registry.get("demo") is _DemoModel
    assert isinstance(registry.create("demo"), _DemoModel)

    resolution = registry.resolve_dependencies("demo")
    assert resolution.name == "demo"
    assert [dep.package_name for dep in resolution.dependencies] == ["numpy", "torch"]

    with pytest.raises(ValueError, match="already registered"):
        registry.register("demo", _DemoModel)

    with pytest.raises(KeyError, match="Unknown model"):
        registry.get("missing")
