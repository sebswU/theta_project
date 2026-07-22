"""Model adapter plugin scaffold for CVPR and research ecosystems.

Each class provides a thin adapter contract over a specific model family.
No algorithmic implementation is provided in this scaffold.
"""

from __future__ import annotations

from typing import Any

from registry.cvpr_model import CVPRModel


class _BaseScaffoldModel(CVPRModel):
    """Shared TODO implementation placeholder for scaffold-only model adapters."""

    def load(self) -> None:
        """Load model resources.

        TODO: Implement model-specific loading logic.
        """
        raise NotImplementedError("TODO: implement load()")

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run model inference.

        TODO: Implement model-specific inference logic.
        """
        raise NotImplementedError("TODO: implement infer()")

    def validate_inputs(self, inputs: dict[str, Any]) -> bool:
        """Validate model inputs.

        TODO: Implement model-specific input validation.
        """
        raise NotImplementedError("TODO: implement validate_inputs()")

    def get_capabilities(self) -> dict[str, Any]:
        """Return model capability metadata.

        TODO: Implement capability declaration.
        """
        raise NotImplementedError("TODO: implement get_capabilities()")

    def get_requirements(self) -> dict[str, Any]:
        """Return model runtime and dependency requirements.

        TODO: Implement requirement declaration.
        """
        raise NotImplementedError("TODO: implement get_requirements()")

    def output_schema(self) -> dict[str, Any]:
        """Return model output schema metadata.

        TODO: Implement schema declaration.
        """
        raise NotImplementedError("TODO: implement output_schema()")


class RTMPoseAdapter(_BaseScaffoldModel):
    """Adapter scaffold for RTMPose."""


class ViTPoseAdapter(_BaseScaffoldModel):
    """Adapter scaffold for ViTPose."""


class SAM2Adapter(_BaseScaffoldModel):
    """Adapter scaffold for SAM2."""


class DUSt3RAdapter(_BaseScaffoldModel):
    """Adapter scaffold for DUSt3R."""


class MASt3RAdapter(_BaseScaffoldModel):
    """Adapter scaffold for MASt3R."""


class MotionBERTAdapter(_BaseScaffoldModel):
    """Adapter scaffold for MotionBERT."""


class GaussianSplattingAdapter(_BaseScaffoldModel):
    """Adapter scaffold for Gaussian Splatting."""


class NeRFAdapter(_BaseScaffoldModel):
    """Adapter scaffold for NeRF."""


class SLAMSystemAdapter(_BaseScaffoldModel):
    """Adapter scaffold for SLAM systems."""
