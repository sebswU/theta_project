"""Tests for fusion plugin scaffold contracts and registry integration."""

import pytest

from fusion import FusionPlugin, FusionPluginLifecycleError, TriangulationPlugin
from registry import PluginRegistry
from schemas import Frame, FusionConfiguration, FusionRequest


def _valid_request() -> FusionRequest:
    return FusionRequest(
        request_id="fusion-1",
        inputs=[Frame(frame_id="f-1", timestamp_ns=1, source_id="cam-1")],
    )


def test_plugin_initialization_occurs_before_processing() -> None:
    """Fusion plugins should enforce initialize() before process()."""
    plugin = TriangulationPlugin()

    with pytest.raises(FusionPluginLifecycleError, match="before initialize"):
        plugin.process(_valid_request())

    plugin.initialize(FusionConfiguration(plugin_name="TriangulationPlugin"))
    response = plugin.process(_valid_request())
    assert response.outputs["route"] == "triangulated_tracks"


def test_input_validation_rejects_malformed_payloads() -> None:
    """Malformed fusion payloads should be rejected at validation/process time."""
    plugin = TriangulationPlugin()
    plugin.initialize(FusionConfiguration(plugin_name="TriangulationPlugin"))

    assert plugin.validate("invalid") is False  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Invalid fusion request"):
        plugin.process(FusionRequest(request_id="fusion-2", inputs=[]))


def test_output_type_routing_is_stable() -> None:
    """Output type should remain deterministic across calls and process outputs."""
    plugin = TriangulationPlugin()
    plugin.initialize(FusionConfiguration(plugin_name="TriangulationPlugin"))

    first = plugin.output_type()
    second = plugin.output_type()
    response = plugin.process(_valid_request())

    assert first == "triangulated_tracks"
    assert second == "triangulated_tracks"
    assert response.outputs["route"] == "triangulated_tracks"
    assert response.metadata["output_type"] == "triangulated_tracks"


def test_plugin_registration_works() -> None:
    """Plugin registry should register, lookup, instantiate, and discover plugins."""
    registry = PluginRegistry()

    registry.register("triangulation", TriangulationPlugin)

    assert registry.discover() == ["triangulation"]
    assert registry.validate("triangulation") is True
    assert registry.get("triangulation") is TriangulationPlugin
    assert isinstance(registry.create("triangulation"), TriangulationPlugin)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("triangulation", TriangulationPlugin)

    with pytest.raises(KeyError, match="Unknown plugin"):
        registry.get("missing")


def test_fusion_base_class_contract_is_enforced() -> None:
    """Subclasses should not override base wrappers directly."""
    with pytest.raises(TypeError, match="Do not override FusionPlugin wrappers"):

        class _BypassFusionPlugin(FusionPlugin):
            def initialize(self, config: FusionConfiguration) -> None:
                return

            def _initialize(self, config: FusionConfiguration) -> None:
                return

            def _process(self, inputs: FusionRequest):
                raise NotImplementedError

            def _validate(self, inputs: FusionRequest) -> bool:
                return True

            def _output_type(self) -> str:
                return "bypass"
