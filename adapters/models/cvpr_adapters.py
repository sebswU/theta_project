"""Model adapter plugin scaffold for CVPR and research ecosystems.

Each class provides a thin adapter contract over a specific model family.
No algorithmic implementation is provided in this scaffold.
"""

from __future__ import annotations

from typing import Any

from registry.cvpr_model import CVPRModel
from schemas.capabilities import SensorType
from schemas.models import (
    InferenceRequest,
    InferenceResponse,
    ModelCapabilities,
    ModelRequirements,
    OutputSchema,
    SceneGraph,
    SceneObject,
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

    def __init__(self) -> None:
        super().__init__()
        self._runtime: Any | None = None
        self._runtime_error: str | None = None

    def _load(self) -> None:
        from adapters.models.rtmpose_onnx import RTMPoseOnnxConfig, RTMPoseOnnxRuntime

        # Keep defaults deterministic while allowing runtime overrides by env/config later.
        config = RTMPoseOnnxConfig(
            onnx_path="registry/rtmpose.onnx",
            device="cpu",
            input_size=(192, 256),
            simcc_split_ratio=2.0,
            use_mmpose_decode=False,
        )
        try:
            runtime = RTMPoseOnnxRuntime(config)
            runtime.load()
            self._runtime = runtime
            self._runtime_error = None
        except Exception as exc:
            # Keep scaffold behavior available for contract tests and minimal runtimes.
            self._runtime = None
            self._runtime_error = str(exc)

    def _infer(self, inputs: InferenceRequest) -> InferenceResponse:
        if self._runtime is None:
            response = super()._infer(inputs)
            if self._runtime_error:
                response.outputs["runtime_unavailable"] = self._runtime_error
            return response

        has_image_payload = all(
            any(key in frame.payload for key in ("image", "image_path", "path", "image_bytes"))
            for frame in inputs.frames
        )
        if not has_image_payload:
            response = super()._infer(inputs)
            response.outputs["runtime_unavailable"] = "frames do not include image payload"
            return response

        pose_results: list[dict[str, Any]] = []
        humans: list[SceneObject] = []
        point_cloud_objects: list[SceneObject] = []
        try:
            for frame in inputs.frames:
                image = self._runtime.decode_image_payload(frame.payload)
                output = self._runtime.infer_image(image)

                frame_people: list[dict[str, Any]] = []
                for person_index, (person_keypoints, person_scores) in enumerate(
                    zip(output["keypoints"], output["scores"], strict=False)
                ):
                    keypoints_2d = person_keypoints.tolist()
                    scores = person_scores.tolist()
                    frame_people.append(
                        {
                            "person_index": person_index,
                            "keypoints": keypoints_2d,
                            "scores": scores,
                        }
                    )
                    humans.append(
                        SceneObject(
                            object_id=(
                                f"{frame.source_id}:{frame.frame_id}:person:{person_index}"
                            ),
                            object_type="person",
                            attributes={
                                "source_id": frame.source_id,
                                "frame_id": frame.frame_id,
                                "person_index": person_index,
                                "keypoints_2d": keypoints_2d,
                                "scores": scores,
                            },
                        )
                    )

                points_xyz = self._extract_points_xyz(frame.payload)
                if points_xyz:
                    point_cloud_objects.append(
                        SceneObject(
                            object_id=f"{frame.source_id}:{frame.frame_id}:point_cloud",
                            object_type="point_cloud",
                            attributes={
                                "source_id": frame.source_id,
                                "frame_id": frame.frame_id,
                                "points_xyz": points_xyz,
                            },
                        )
                    )

                pose_results.append(
                    {
                        "frame_id": frame.frame_id,
                        "source_id": frame.source_id,
                        "people": frame_people,
                        "timings_ms": output["timings_ms"],
                    }
                )
        except Exception as exc:
            response = super()._infer(inputs)
            response.outputs["runtime_unavailable"] = str(exc)
            return response

        return InferenceResponse(
            request_id=inputs.request_id,
            outputs={
                "model_name": self.model_name,
                "poses_2d": pose_results,
                "frame_count": len(pose_results),
                "point_cloud_count": len(point_cloud_objects),
            },
            scene_graph=SceneGraph(
                scene_id=f"rtmpose:{inputs.request_id}",
                humans=humans,
                objects=point_cloud_objects,
            ),
        )

    def _get_requirements(self) -> ModelRequirements:
        return ModelRequirements(
            model_name=self.model_name,
            required_sources=list(self.REQUIRED_SOURCES),
            resources={
                "dependencies": [
                    "numpy>=2.0.0",
                    "opencv-python>=4.8.0",
                    "onnxruntime>=1.8.1",
                ],
                "artifacts": ["registry/rtmpose.onnx"],
            },
            metadata={"adapter_class": self.__class__.__name__, "supports_mmpose_decode": False},
        )

    def _output_schema(self) -> OutputSchema:
        return OutputSchema(
            name="poses_2d",
            fields=["frame_id", "source_id", "people", "timings_ms"],
        )

    @staticmethod
    def _extract_points_xyz(payload: dict[str, Any]) -> list[list[float]]:
        """Normalize common point-cloud payload variants into xyz float triplets."""
        raw_points = payload.get("points_xyz", payload.get("point_cloud"))
        if raw_points is None:
            return []

        if hasattr(raw_points, "tolist"):
            raw_points = raw_points.tolist()

        if not isinstance(raw_points, list):
            return []

        points_xyz: list[list[float]] = []
        for point in raw_points:
            if not isinstance(point, (list, tuple)) or len(point) < 3:
                continue
            try:
                points_xyz.append([float(point[0]), float(point[1]), float(point[2])])
            except (TypeError, ValueError):
                continue
        return points_xyz


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
