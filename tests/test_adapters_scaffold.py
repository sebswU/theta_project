"""Tests for adapter lifecycle and contract scaffolding."""

import pytest

from adapters import SensorAdapter, SensorAdapterLifecycleError, URISensorAdapter
from schemas.capabilities import SensorType
from schemas.models import Frame, SourceDescriptor


class _RecorderAdapter(SensorAdapter):
    """Test adapter that records lifecycle hook calls."""

    def __init__(self, source: SourceDescriptor) -> None:
        super().__init__(source)
        self.events: list[str] = []

    def _connect(self) -> None:
        self.events.append("connect")

    def _read(self) -> Frame:
        self.events.append("read")
        return Frame(frame_id="f-1", timestamp_ns=1, source_id=self.source.source_id)

    def _disconnect(self) -> None:
        self.events.append("disconnect")


def _source_descriptor() -> SourceDescriptor:
    return SourceDescriptor(source_id="cam-1", source_type=SensorType.RGB_CAMERA)


def test_lifecycle_ordering_is_enforced() -> None:
    """Adapters should require connect before read, then allow disconnect."""
    adapter = _RecorderAdapter(_source_descriptor())

    adapter.connect()
    frame = adapter.read()
    adapter.disconnect()

    assert frame.source_id == "cam-1"
    assert adapter.events == ["connect", "read", "disconnect"]
    assert adapter.is_connected is False


def test_read_before_connect_fails_clearly() -> None:
    """read() should fail with an explicit lifecycle error before connect()."""
    adapter = _RecorderAdapter(_source_descriptor())

    with pytest.raises(SensorAdapterLifecycleError, match="before connect"):
        adapter.read()


def test_disconnect_is_idempotent() -> None:
    """Calling disconnect() multiple times should be safe."""
    adapter = _RecorderAdapter(_source_descriptor())

    adapter.connect()
    adapter.disconnect()
    adapter.disconnect()

    assert adapter.events == ["connect", "disconnect"]
    assert adapter.is_connected is False


def test_abstract_contract_cannot_be_bypassed() -> None:
    """Subclasses should not override lifecycle wrappers directly."""
    with pytest.raises(TypeError, match="Do not override lifecycle wrappers"):

        class _BypassAdapter(SensorAdapter):
            def connect(self) -> None:
                return

            def _connect(self) -> None:
                return

            def _read(self) -> Frame:
                return Frame(frame_id="f-1", timestamp_ns=1, source_id="cam-1")

            def _disconnect(self) -> None:
                return


def test_uri_sensor_adapter_emits_uri_in_frame_payload() -> None:
    """URI-backed adapter should connect and surface URI context in frame payload."""
    source = SourceDescriptor(
        source_id="stream-1",
        source_type=SensorType.NETWORK_STREAM,
        uri="rtsp://localhost:8554/live",
    )
    adapter = URISensorAdapter(source)

    adapter.connect()
    frame = adapter.read()
    adapter.disconnect()

    assert frame.source_id == "stream-1"
    assert frame.payload["uri"] == "rtsp://localhost:8554/live"
    assert frame.payload["connection_kind"] == "network"
    assert frame.payload["sequence"] == 1


def test_uri_sensor_adapter_rejects_missing_uri() -> None:
    """URI-backed adapter should fail fast when source URI is absent."""
    source = SourceDescriptor(
        source_id="stream-2",
        source_type=SensorType.NETWORK_STREAM,
        uri=None,
    )
    adapter = URISensorAdapter(source)

    with pytest.raises(ValueError, match="Source URI is required"):
        adapter.connect()


def test_uri_sensor_adapter_supports_local_path_uri() -> None:
    """Plain local paths should be treated as path-style connection endpoints."""
    source = SourceDescriptor(
        source_id="dataset-1",
        source_type=SensorType.DATASET,
        uri="./data/sample.mp4",
    )
    adapter = URISensorAdapter(source)

    adapter.connect()
    frame = adapter.read()
    adapter.disconnect()

    assert frame.payload["connection_kind"] == "path"
    assert frame.payload["uri"].endswith("data/sample.mp4")
