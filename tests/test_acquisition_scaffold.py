"""Tests for acquisition discovery and capability detection scaffolding."""

import pytest

from acquisition import BaseDiscoveryProvider, CapabilityDetector
from schemas.capabilities import SensorCapabilityProfile, SensorType
from schemas.models import SourceDescriptor


class _MixedDiscoveryProvider(BaseDiscoveryProvider):
    """Discovery provider that emits mixed source descriptor shapes."""

    def _discover(self):
        return [
            {"source_id": "cam-b", "source_type": "depth_camera", "metadata": {"rank": 2}},
            SourceDescriptor(
                source_id="cam-a",
                source_type=SensorType.RGB_CAMERA,
                metadata={"rank": 1},
            ),
        ]


class _BadDiscoveryProvider(BaseDiscoveryProvider):
    """Discovery provider that emits an unsupported payload shape."""

    def _discover(self):
        return [123]


class _TypedCapabilityDetector(CapabilityDetector):
    """Capability detector that returns a partially specified profile payload."""

    def _detect(self, source_descriptor: SourceDescriptor):
        return {"supports_calibration": True, "supports_multiview": True}


class _BadCapabilityDetector(CapabilityDetector):
    """Capability detector that emits an unsupported payload shape."""

    def _detect(self, source_descriptor: SourceDescriptor):
        return 42


def test_source_discovery_returns_normalized_descriptors() -> None:
    """Discovery should return sorted canonical SourceDescriptor instances."""
    provider = _MixedDiscoveryProvider()

    discovered = provider.discover()

    assert [item.source_id for item in discovered] == ["cam-a", "cam-b"]
    assert all(isinstance(item, SourceDescriptor) for item in discovered)
    assert discovered[0].source_type is SensorType.RGB_CAMERA
    assert discovered[1].source_type is SensorType.DEPTH_CAMERA


def test_unknown_source_shapes_fail_safely() -> None:
    """Unsupported discovery payloads should fail with a clear error."""
    provider = _BadDiscoveryProvider()

    with pytest.raises(TypeError, match="Unsupported source descriptor shape"):
        provider.discover()


def test_capability_detection_maps_descriptors_into_typed_profiles() -> None:
    """Capability detection should preserve descriptor identity and type."""
    detector = _TypedCapabilityDetector()
    source = SourceDescriptor(source_id="cam-1", source_type=SensorType.RGB_CAMERA)

    profile = detector.detect(source)

    assert isinstance(profile, SensorCapabilityProfile)
    assert profile.source_id == "cam-1"
    assert profile.sensor_type is SensorType.RGB_CAMERA
    assert profile.supports_calibration is True
    assert profile.supports_multiview is True


def test_unknown_capability_shapes_fail_safely() -> None:
    """Unsupported capability payloads should fail with a clear error."""
    detector = _BadCapabilityDetector()
    source = SourceDescriptor(source_id="cam-1", source_type=SensorType.RGB_CAMERA)

    with pytest.raises(TypeError, match="Unsupported capability profile shape"):
        detector.detect(source)


def test_boundary_contracts_cannot_be_bypassed() -> None:
    """Subclasses should not override public discovery or detection wrappers."""
    with pytest.raises(TypeError, match="Do not override discovery wrapper"):

        class _BypassDiscovery(BaseDiscoveryProvider):
            def discover(self):
                return []

            def _discover(self):
                return []

    with pytest.raises(TypeError, match="Do not override detection wrapper"):

        class _BypassDetection(CapabilityDetector):
            def detect(self, source_descriptor: SourceDescriptor):
                return SensorCapabilityProfile(source_id="x", sensor_type=SensorType.RGB_CAMERA)

            def _detect(self, source_descriptor: SourceDescriptor):
                return {}
