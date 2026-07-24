"""Fusion plugin scaffold definitions.

All fusion plugins must inherit from `FusionPlugin` and implement required
lifecycle and processing contracts.
"""

from __future__ import annotations

from fusion.base import FusionPlugin
from schemas.models import FusionConfiguration
from schemas.models import FusionRequest
from schemas.models import FusionResponse


class _BaseScaffoldFusionPlugin(FusionPlugin):
    """Shared placeholder implementation for scaffold-only fusion plugins."""

    def initialize(self, config: FusionConfiguration) -> None:
        """Initialize plugin.

        TODO: Implement plugin-specific initialization.
        """
        raise NotImplementedError("TODO: implement initialize()")

    def process(self, inputs: FusionRequest) -> FusionResponse:
        """Process fusion inputs.

        TODO: Implement plugin-specific fusion logic.
        """
        raise NotImplementedError("TODO: implement process()")

    def validate(self, inputs: FusionRequest) -> bool:
        """Validate plugin inputs.

        TODO: Implement plugin input validation.
        """
        raise NotImplementedError("TODO: implement validate()")

    def output_type(self) -> str:
        """Return plugin output semantic type.

        TODO: Implement output type declaration.
        """
        raise NotImplementedError("TODO: implement output_type()")


class TriangulationPlugin(_BaseScaffoldFusionPlugin):
    """Triangulation fusion plugin scaffold."""


class BundleAdjustmentPlugin(_BaseScaffoldFusionPlugin):
    """Bundle adjustment fusion plugin scaffold."""


class CrossViewMatchingPlugin(_BaseScaffoldFusionPlugin):
    """Cross-view matching fusion plugin scaffold."""


class TemporalFusionPlugin(_BaseScaffoldFusionPlugin):
    """Temporal fusion plugin scaffold."""


class VolumetricFusionPlugin(_BaseScaffoldFusionPlugin):
    """Volumetric fusion plugin scaffold."""


class NeuralFusionPlugin(_BaseScaffoldFusionPlugin):
    """Neural fusion plugin scaffold."""


class SceneGraphPlugin(_BaseScaffoldFusionPlugin):
    """Scene graph fusion plugin scaffold."""
