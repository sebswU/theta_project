"""Tests for adapter lifecycle and contract scaffolding."""

import pytest

from adapters import (
    SensorAdapter,
    SensorAdapterLifecycleError,
    SkellyCamFrameAdapter,
    URISensorAdapter,
)
from adapters.skelly_api_bridge import _decode_binary_payload
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


def test_skellycam_frame_adapter_outputs_canonical_frame() -> None:
    """SkellyCam frames should map into the canonical Frame envelope."""
    source = _source_descriptor()
    adapter = SkellyCamFrameAdapter(
        source,
        frames=[
            {
                "frame_id": "raw-1",
                "source_id": "cam-1",
                "timestamp_ns": 123,
                "payload": {"image": "rgb-bytes"},
            }
        ],
    )

    adapter.connect()
    frame = adapter.read()
    adapter.disconnect()

    assert isinstance(frame, Frame)
    assert frame.frame_id == "raw-1"
    assert frame.source_id == "cam-1"
    assert frame.timestamp_ns == 123
    assert frame.payload["image"] == "rgb-bytes"


def test_skellycam_frame_adapter_rejects_missing_timestamp() -> None:
    """Frames missing timestamp fields should fail safe."""
    source = _source_descriptor()
    adapter = SkellyCamFrameAdapter(
        source,
        frames=[{"frame_id": "raw-2", "source_id": "cam-1", "payload": {}}],
    )

    adapter.connect()
    with pytest.raises(ValueError, match="missing required timestamp"):
        adapter.read()


def test_skellycam_frame_adapter_rejects_missing_source_id() -> None:
    """Frames missing source identity should fail safe."""
    source = _source_descriptor()
    adapter = SkellyCamFrameAdapter(
        source,
        frames=[{"frame_id": "raw-3", "timestamp_ns": 456, "payload": {}}],
    )

    adapter.connect()
    with pytest.raises(ValueError, match="missing required source_id"):
        adapter.read()


def test_skellycam_frame_adapter_preserves_payload_metadata() -> None:
    """Metadata from incoming payload should survive canonical wrapping."""
    source = _source_descriptor()
    adapter = SkellyCamFrameAdapter(
        source,
        frames=[
            {
                "id": "fallback-id",
                "source_id": "cam-1",
                "timestamp": 789,
                "metadata": {"exposure": 0.01, "gain": 2.0},
                "payload": {"encoding": "rgb8"},
            }
        ],
    )

    adapter.connect()
    frame = adapter.read()
    adapter.disconnect()

    assert frame.timestamp_ns == 789
    assert frame.payload["encoding"] == "rgb8"
    assert frame.payload["metadata"] == {"exposure": 0.01, "gain": 2.0}


def test_skelly_api_bridge_parses_aligned_little_endian_binary_layout() -> None:
    """Bridge should parse 24/56/N-JPEG/24 aligned little-endian websocket payloads."""
    np = pytest.importorskip("numpy")

    payload_header_dtype = np.dtype(
        [
            ("message_type", "u1"),
            ("_pad0", "V7"),
            ("frame_number", "<i8"),
            ("number_of_cameras", "<i4"),
            ("_pad1", "V4"),
        ],
        align=True,
    )
    frame_header_dtype = np.dtype(
        [
            ("message_type", "u1"),
            ("_pad0", "V7"),
            ("frame_number", "<i8"),
            ("camera_id", "S16"),
            ("camera_index", "<i4"),
            ("image_width", "<i4"),
            ("image_height", "<i4"),
            ("color_channels", "<i4"),
            ("jpeg_string_length", "<i4"),
            ("_pad1", "V4"),
        ],
        align=True,
    )
    payload_footer_dtype = np.dtype(
        [
            ("message_type", "u1"),
            ("_pad0", "V7"),
            ("frame_number", "<i8"),
            ("number_of_cameras", "<i4"),
            ("_pad1", "V4"),
        ],
        align=True,
    )

    header = np.zeros((), dtype=payload_header_dtype)
    header["message_type"] = 0
    header["frame_number"] = 42
    header["number_of_cameras"] = 2
    header_bytes = header.tobytes()

    jpeg0 = b"\xff\xd8\xff\xdbcam0\xff\xd9"
    jpeg1 = b"\xff\xd8\xff\xdbcam1-stream\xff\xd9"

    frame0 = np.zeros((), dtype=frame_header_dtype)
    frame0["message_type"] = 1
    frame0["frame_number"] = 42
    frame0["camera_id"] = b"front_rgb\x00"
    frame0["camera_index"] = 0
    frame0["image_width"] = 1280
    frame0["image_height"] = 720
    frame0["color_channels"] = 3
    frame0["jpeg_string_length"] = len(jpeg0)

    frame1 = np.zeros((), dtype=frame_header_dtype)
    frame1["message_type"] = 1
    frame1["frame_number"] = 42
    frame1["camera_id"] = b"left_depth\x00"
    frame1["camera_index"] = 1
    frame1["image_width"] = 640
    frame1["image_height"] = 480
    frame1["color_channels"] = 3
    frame1["jpeg_string_length"] = len(jpeg1)

    footer = np.zeros((), dtype=payload_footer_dtype)
    footer["message_type"] = 2
    footer["frame_number"] = 42
    footer["number_of_cameras"] = 2

    payload_bytes = b"".join(
        [
            header_bytes,
            frame0.tobytes(),
            jpeg0,
            frame1.tobytes(),
            jpeg1,
            footer.tobytes(),
        ]
    )

    parsed = _decode_binary_payload(payload_bytes)

    assert parsed["frame_id"] == "frame-42"
    binary_payload = parsed["payload"]
    assert binary_payload["layout"]["endianness"] == "little"
    assert binary_payload["layout"]["alignment"] == "numpy.align=True"
    assert binary_payload["payload_header"]["message_type"] == 0
    assert binary_payload["payload_header"]["frame_number"] == 42
    assert binary_payload["payload_header"]["number_of_cameras"] == 2

    cameras = binary_payload["cameras"]
    assert len(cameras) == 2
    assert cameras[0]["frame_header"]["message_type"] == 1
    assert cameras[0]["frame_header"]["frame_number"] == 42
    assert cameras[0]["frame_header"]["camera_id"] == "front_rgb"
    assert cameras[0]["frame_header"]["camera_index"] == 0
    assert cameras[0]["frame_header"]["jpeg_string_length"] == len(jpeg0)
    assert cameras[0]["jpeg"] == jpeg0
    assert cameras[1]["frame_header"]["message_type"] == 1
    assert cameras[1]["frame_header"]["frame_number"] == 42
    assert cameras[1]["frame_header"]["camera_id"] == "left_depth"
    assert cameras[1]["frame_header"]["camera_index"] == 1
    assert cameras[1]["frame_header"]["jpeg_string_length"] == len(jpeg1)
    assert cameras[1]["jpeg"] == jpeg1
    assert binary_payload["payload_footer"]["message_type"] == 2
    assert binary_payload["payload_footer"]["frame_number"] == 42
    assert binary_payload["payload_footer"]["number_of_cameras"] == 2


def test_skelly_api_bridge_rejects_truncated_binary_payload() -> None:
    """Bridge should reject incomplete binary payloads with clear errors."""
    np = pytest.importorskip("numpy")

    payload_header_dtype = np.dtype(
        [
            ("message_type", "u1"),
            ("_pad0", "V7"),
            ("frame_number", "<i8"),
            ("number_of_cameras", "<i4"),
            ("_pad1", "V4"),
        ],
        align=True,
    )
    frame_header_dtype = np.dtype(
        [
            ("message_type", "u1"),
            ("_pad0", "V7"),
            ("frame_number", "<i8"),
            ("camera_id", "S16"),
            ("camera_index", "<i4"),
            ("image_width", "<i4"),
            ("image_height", "<i4"),
            ("color_channels", "<i4"),
            ("jpeg_string_length", "<i4"),
            ("_pad1", "V4"),
        ],
        align=True,
    )
    payload_footer_dtype = np.dtype(
        [
            ("message_type", "u1"),
            ("_pad0", "V7"),
            ("frame_number", "<i8"),
            ("number_of_cameras", "<i4"),
            ("_pad1", "V4"),
        ],
        align=True,
    )

    header = np.zeros((), dtype=payload_header_dtype)
    header["message_type"] = 0
    header["frame_number"] = 1
    header["number_of_cameras"] = 1

    frame = np.zeros((), dtype=frame_header_dtype)
    frame["message_type"] = 1
    frame["frame_number"] = 1
    frame["camera_id"] = b"cam\x00"
    frame["camera_index"] = 0
    frame["image_width"] = 640
    frame["image_height"] = 480
    frame["color_channels"] = 3
    frame["jpeg_string_length"] = 10

    footer = np.zeros((), dtype=payload_footer_dtype)
    footer["message_type"] = 2
    footer["frame_number"] = 1
    footer["number_of_cameras"] = 1
    payload_bytes = b"".join([header.tobytes(), frame.tobytes(), b"\x00\x01", footer.tobytes()])

    with pytest.raises(ValueError, match="JPEG bytes were fully read"):
        _decode_binary_payload(payload_bytes)


def test_skelly_api_bridge_rejects_invalid_payload_header_message_type() -> None:
    """Bridge should reject binary payloads when header message_type is not PAYLOAD_HEADER."""
    np = pytest.importorskip("numpy")

    payload_header_dtype = np.dtype(
        [
            ("message_type", "u1"),
            ("_pad0", "V7"),
            ("frame_number", "<i8"),
            ("number_of_cameras", "<i4"),
            ("_pad1", "V4"),
        ],
        align=True,
    )
    payload_footer_dtype = np.dtype(
        [
            ("message_type", "u1"),
            ("_pad0", "V7"),
            ("frame_number", "<i8"),
            ("number_of_cameras", "<i4"),
            ("_pad1", "V4"),
        ],
        align=True,
    )

    header = np.zeros((), dtype=payload_header_dtype)
    header["message_type"] = 9
    header["frame_number"] = 7
    header["number_of_cameras"] = 0

    footer = np.zeros((), dtype=payload_footer_dtype)
    footer["message_type"] = 2
    footer["frame_number"] = 7
    footer["number_of_cameras"] = 0

    with pytest.raises(ValueError, match="header message_type mismatch"):
        _decode_binary_payload(header.tobytes() + footer.tobytes())


def test_skelly_api_bridge_rejects_invalid_frame_header_message_type() -> None:
    """Bridge should reject binary payloads with invalid FRAME_HEADER message_type."""
    np = pytest.importorskip("numpy")

    payload_header_dtype = np.dtype(
        [
            ("message_type", "u1"),
            ("_pad0", "V7"),
            ("frame_number", "<i8"),
            ("number_of_cameras", "<i4"),
            ("_pad1", "V4"),
        ],
        align=True,
    )
    frame_header_dtype = np.dtype(
        [
            ("message_type", "u1"),
            ("_pad0", "V7"),
            ("frame_number", "<i8"),
            ("camera_id", "S16"),
            ("camera_index", "<i4"),
            ("image_width", "<i4"),
            ("image_height", "<i4"),
            ("color_channels", "<i4"),
            ("jpeg_string_length", "<i4"),
            ("_pad1", "V4"),
        ],
        align=True,
    )
    payload_footer_dtype = np.dtype(
        [
            ("message_type", "u1"),
            ("_pad0", "V7"),
            ("frame_number", "<i8"),
            ("number_of_cameras", "<i4"),
            ("_pad1", "V4"),
        ],
        align=True,
    )

    header = np.zeros((), dtype=payload_header_dtype)
    header["message_type"] = 0
    header["frame_number"] = 10
    header["number_of_cameras"] = 1

    frame = np.zeros((), dtype=frame_header_dtype)
    frame["message_type"] = 9
    frame["frame_number"] = 10
    frame["camera_id"] = b"front_rgb\x00"
    frame["camera_index"] = 0
    frame["image_width"] = 1280
    frame["image_height"] = 720
    frame["color_channels"] = 3
    frame["jpeg_string_length"] = 3

    footer = np.zeros((), dtype=payload_footer_dtype)
    footer["message_type"] = 2
    footer["frame_number"] = 10
    footer["number_of_cameras"] = 1

    payload_bytes = b"".join([header.tobytes(), frame.tobytes(), b"abc", footer.tobytes()])

    with pytest.raises(ValueError, match="frame header message_type mismatch"):
        _decode_binary_payload(payload_bytes)
