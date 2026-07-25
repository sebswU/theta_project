"""Model adapter plugin scaffold for CVPR and research ecosystems.

Each class provides a thin adapter contract over a specific model family.
No algorithmic implementation is provided in this scaffold.
"""

from __future__ import annotations

from registry.cvpr_model import CVPRModel
from schemas.capabilities import SensorType
from schemas.models import (
    InferenceRequest,
    InferenceResponse,
    ModelCapabilities,
    ModelRequirements,
    OutputSchema,
)


class _BaseScaffoldModel(CVPRModel):
    """Shared TODO implementation placeholder for scaffold-only model adapters."""

    MODEL_NAME = "base-scaffold-model"
    SUPPORTED_SOURCES: list[SensorType] = [SensorType.RGB_CAMERA]
    REQUIRED_SOURCES: list[SensorType] = [SensorType.RGB_CAMERA]
    OUTPUT_TYPES: list[str] = ["detections"]

    @property
    def model_name(self) -> str:
        return self.MODEL_NAME

    def _load(self) -> None:
        """Load model resources.

        Scaffold models do not load external artifacts yet.
        """

    def _infer(self, inputs: InferenceRequest) -> InferenceResponse:
        """Run model inference.

        Scaffold models return a deterministic envelope for registry/config wiring.
        """
        return InferenceResponse(
            request_id=inputs.request_id,
            outputs={
                "model_name": self.model_name,
                "frame_count": len(inputs.frames),
                "output_types": list(self.OUTPUT_TYPES),
            },
        )

    def _validate_inputs(self, inputs: InferenceRequest) -> bool:
        """Validate model inputs."""
        return all(frame.source_id.strip() for frame in inputs.frames)

    def _get_capabilities(self) -> ModelCapabilities:
        """Return model capability metadata."""
        return ModelCapabilities(
            model_name=self.model_name,
            supported_sources=list(self.SUPPORTED_SOURCES),
            output_types=list(self.OUTPUT_TYPES),
            metadata={"adapter_class": self.__class__.__name__},
        )

    def _get_requirements(self) -> ModelRequirements:
        """Return model runtime and dependency requirements."""
        return ModelRequirements(
            model_name=self.model_name,
            required_sources=list(self.REQUIRED_SOURCES),
            resources={"dependencies": []},
            metadata={"adapter_class": self.__class__.__name__},
        )

    def _output_schema(self) -> OutputSchema:
        """Return model output schema metadata."""
        primary_output = self.OUTPUT_TYPES[0] if self.OUTPUT_TYPES else "application_output"
        return OutputSchema(name=primary_output, fields=["model_name", "frame_count"])


class RTMPoseAdapter(_BaseScaffoldModel):
    """Adapter scaffold for RTMPose."""

    MODEL_NAME = "rtmpose"
    OUTPUT_TYPES = ["poses_2d"]


class ViTPoseAdapter(_BaseScaffoldModel):
    """Adapter scaffold for ViTPose."""

    MODEL_NAME = "vitpose"
    OUTPUT_TYPES = ["poses_2d"]


class SAM2Adapter(_BaseScaffoldModel):
    """Adapter scaffold for SAM2."""

    MODEL_NAME = "sam2"
    OUTPUT_TYPES = ["masks"]


class DUSt3RAdapter(_BaseScaffoldModel):
    """Adapter scaffold for DUSt3R."""

    MODEL_NAME = "dust3r"
    SUPPORTED_SOURCES = [SensorType.RGB_CAMERA, SensorType.DATASET]
    REQUIRED_SOURCES = [SensorType.RGB_CAMERA]
    OUTPUT_TYPES = ["point_cloud"]


class MASt3RAdapter(_BaseScaffoldModel):
    """Adapter scaffold for MASt3R."""

    MODEL_NAME = "mast3r"
    SUPPORTED_SOURCES = [SensorType.RGB_CAMERA, SensorType.DATASET]
    REQUIRED_SOURCES = [SensorType.RGB_CAMERA]
    OUTPUT_TYPES = ["point_cloud"]


class MotionBERTAdapter(_BaseScaffoldModel):
    """Adapter scaffold for MotionBERT."""

    MODEL_NAME = "motionbert"
    OUTPUT_TYPES = ["poses_3d"]


class GaussianSplattingAdapter(_BaseScaffoldModel):
    """Adapter scaffold for Gaussian Splatting."""

    MODEL_NAME = "gaussian_splatting"
    SUPPORTED_SOURCES = [SensorType.RGB_CAMERA, SensorType.DATASET]
    REQUIRED_SOURCES = [SensorType.RGB_CAMERA]
    OUTPUT_TYPES = ["gaussian_scene"]


class NeRFAdapter(_BaseScaffoldModel):
    """Adapter scaffold for NeRF."""

    MODEL_NAME = "nerf"
    SUPPORTED_SOURCES = [SensorType.RGB_CAMERA, SensorType.DATASET]
    REQUIRED_SOURCES = [SensorType.RGB_CAMERA]
    OUTPUT_TYPES = ["radiance_field"]


class SLAMSystemAdapter(_BaseScaffoldModel):
    """Adapter scaffold for SLAM systems."""

    MODEL_NAME = "slam_system"
    SUPPORTED_SOURCES = [SensorType.RGB_CAMERA, SensorType.DEPTH_CAMERA, SensorType.IMU]
    REQUIRED_SOURCES = [SensorType.RGB_CAMERA]
    OUTPUT_TYPES = ["trajectory"]


__all__ = [
    "DUSt3RAdapter",
    "GaussianSplattingAdapter",
    "MASt3RAdapter",
    "MotionBERTAdapter",
    "NeRFAdapter",
    "RTMPoseAdapter",
    "SAM2Adapter",
    "SLAMSystemAdapter",
    "ViTPoseAdapter",
]
