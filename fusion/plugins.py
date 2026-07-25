"""Fusion plugin scaffold definitions.

All fusion plugins must inherit from `FusionPlugin` and implement required
lifecycle and processing contracts.
"""

from __future__ import annotations

from fusion.base import FusionPlugin
from schemas.models import FusionConfiguration, FusionRequest, FusionResponse


class _BaseScaffoldFusionPlugin(FusionPlugin):
    """Shared placeholder implementation for scaffold-only fusion plugins."""

    OUTPUT_TYPE = "generic_fusion"

    def _initialize(self, config: FusionConfiguration) -> None:
        """Initialize plugin.

        Scaffold implementation records configuration through base wrapper.
        """
        if config.plugin_name != self.__class__.__name__:
            raise ValueError(
                f"Fusion config plugin_name '{config.plugin_name}' does not match "
                f"plugin class '{self.__class__.__name__}'"
            )

    def _process(self, inputs: FusionRequest) -> FusionResponse:
        """Process fusion inputs.

        Scaffold output summarizes the normalized input for downstream wiring.
        """
        route = self.output_type()
        return FusionResponse(
            request_id=inputs.request_id,
            outputs={
                "route": route,
                "frame_count": len(inputs.inputs),
                "has_scene_graph": inputs.scene_graph is not None,
            },
            scene_graph=inputs.scene_graph,
            metadata={"plugin_name": self.config.plugin_name, "output_type": route},
        )

    def _validate(self, inputs: FusionRequest) -> bool:
        """Validate plugin inputs.

        Scaffold plugins require at least one frame as normalized input.
        """
        return len(inputs.inputs) > 0

    def _output_type(self) -> str:
        """Return plugin output semantic type.

        Output type is a stable class-level constant for deterministic routing.
        """
        return self.OUTPUT_TYPE


class TriangulationPlugin(_BaseScaffoldFusionPlugin):
    """Triangulation fusion plugin scaffold."""

    OUTPUT_TYPE = "triangulated_tracks"


class BundleAdjustmentPlugin(_BaseScaffoldFusionPlugin):
    """Bundle adjustment fusion plugin scaffold."""

    OUTPUT_TYPE = "bundle_adjusted_scene"


class CrossViewMatchingPlugin(_BaseScaffoldFusionPlugin):
    """Cross-view matching fusion plugin scaffold."""

    OUTPUT_TYPE = "cross_view_matches"


class TemporalFusionPlugin(_BaseScaffoldFusionPlugin):
    """Temporal fusion plugin scaffold."""

    OUTPUT_TYPE = "temporal_fusion_sequence"


class VolumetricFusionPlugin(_BaseScaffoldFusionPlugin):
    """Volumetric fusion plugin scaffold."""

    OUTPUT_TYPE = "volumetric_field"


class NeuralFusionPlugin(_BaseScaffoldFusionPlugin):
    """Neural fusion plugin scaffold."""

    OUTPUT_TYPE = "neural_scene_representation"


class SceneGraphPlugin(_BaseScaffoldFusionPlugin):
    """Scene graph fusion plugin scaffold."""

    OUTPUT_TYPE = "scene_graph"


__all__ = [
    "BundleAdjustmentPlugin",
    "CrossViewMatchingPlugin",
    "NeuralFusionPlugin",
    "SceneGraphPlugin",
    "TemporalFusionPlugin",
    "TriangulationPlugin",
    "VolumetricFusionPlugin",
]
