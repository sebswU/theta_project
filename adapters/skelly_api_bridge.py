"""Skelly websocket bridge adapter.

This adapter connects to a websocket endpoint and converts incoming payloads
into canonical ``schemas.models.Frame`` objects.

Supported payload modes:
- JSON object messages.
- Little-endian binary payloads with aligned (numpy ``align=True``) layout:
    payload header (24 bytes), N * (frame header (56 bytes) + JPEG bytes),
    payload footer (24 bytes).
"""

from __future__ import annotations

import json
from time import time_ns
from typing import Any

from adapters.sensor_adapter import SensorAdapter, SensorAdapterLifecycleError
from schemas.models import Frame

_PAYLOAD_HEADER_SIZE = 24
_FRAME_HEADER_SIZE = 56
_PAYLOAD_FOOTER_SIZE = 24
_PAYLOAD_HEADER_TYPE = 0
_FRAME_HEADER_TYPE = 1
_PAYLOAD_FOOTER_TYPE = 2


class SkellyApiBridgeAdapter(SensorAdapter):
    """Read SkellyCam frames from a websocket endpoint.

    The adapter expects the bound source URI to be a websocket URL, e.g.
    ``ws://localhost:53117/skellycam/websocket``.
    """

    def __init__(self, source) -> None:
        super().__init__(source)
        self._socket = None

    def _connect(self) -> None:
        uri = (self.source.uri or "").strip()
        if not uri:
            raise ValueError("Source URI is required for websocket bridge")
        if not uri.startswith(("ws://", "wss://")):
            raise ValueError("Websocket bridge requires ws:// or wss:// source URI")

        try:
            from websocket import create_connection  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "websocket-client is required for SkellyApiBridgeAdapter. "
                "Install with: python3 -m pip install websocket-client"
            ) from exc

        self._socket = create_connection(uri)

    def _read(self) -> Frame:
        if self._socket is None:
            raise SensorAdapterLifecycleError("Skelly websocket bridge is not connected")

        message = self._socket.recv()
        payload = _decode_message(message)
        return _normalize_payload(payload, expected_source_id=self.source.source_id)

    def _disconnect(self) -> None:
        if self._socket is None:
            return
        try:
            self._socket.close()
        finally:
            self._socket = None


def _decode_message(message: Any) -> dict[str, Any]:
    if isinstance(message, bytes):
        stripped = message.lstrip()
        if stripped.startswith((b"{", b"[")):
            message = message.decode("utf-8")
        else:
            return _decode_binary_payload(message)

    if isinstance(message, str):
        try:
            parsed = json.loads(message)
        except json.JSONDecodeError as exc:
            raise ValueError("Websocket message is not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Websocket message JSON must be an object")
        return parsed

    if isinstance(message, dict):
        return message

    raise TypeError("Unsupported websocket payload shape")


def _normalize_payload(payload: dict[str, Any], *, expected_source_id: str) -> Frame:
    source_id_value = payload.get("source_id", expected_source_id)
    if not isinstance(source_id_value, str) or not source_id_value.strip():
        raise ValueError("Websocket frame is missing source_id")
    source_id = source_id_value.strip()
    if source_id != expected_source_id:
        raise ValueError("Websocket frame source_id does not match adapter source")

    ts_value = payload.get("timestamp_ns", payload.get("timestamp", time_ns()))
    if isinstance(ts_value, bool) or not isinstance(ts_value, (int, float)):
        raise ValueError("Websocket frame timestamp must be numeric")
    timestamp_ns = int(ts_value)

    frame_id_value = payload.get("frame_id") or payload.get("id")
    if isinstance(frame_id_value, str) and frame_id_value.strip():
        frame_id = frame_id_value.strip()
    else:
        frame_id = f"{source_id}-{timestamp_ns}"

    frame_payload = payload.get("payload")
    if frame_payload is None:
        frame_payload = {}
    elif not isinstance(frame_payload, dict):
        raise ValueError("Websocket frame payload must be an object")
    else:
        frame_payload = dict(frame_payload)

    passthrough = {
        key: value
        for key, value in payload.items()
        if key not in {"id", "frame_id", "source_id", "timestamp", "timestamp_ns", "payload"}
    }
    frame_payload.update(passthrough)

    return Frame(
        frame_id=frame_id,
        timestamp_ns=timestamp_ns,
        source_id=source_id,
        payload=frame_payload,
    )


def _decode_binary_payload(message: bytes) -> dict[str, Any]:
    """Parse aligned little-endian Skelly binary websocket payloads.

    Layout:
    - payload header: 24 bytes
    - repeated camera blocks: frame header (56 bytes) + JPEG bytes
    - payload footer: 24 bytes
    """
    if len(message) < (_PAYLOAD_HEADER_SIZE + _PAYLOAD_FOOTER_SIZE):
        raise ValueError("Binary websocket payload is too small")

    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "numpy is required for aligned binary payload parsing. "
            "Install with: python3 -m pip install numpy"
        ) from exc

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

    if payload_header_dtype.itemsize != _PAYLOAD_HEADER_SIZE:
        raise ValueError("Internal payload header dtype mismatch")
    if frame_header_dtype.itemsize != _FRAME_HEADER_SIZE:
        raise ValueError("Internal frame header dtype mismatch")
    if payload_footer_dtype.itemsize != _PAYLOAD_FOOTER_SIZE:
        raise ValueError("Internal payload footer dtype mismatch")

    # parse payload header
    # cursor is used to track the current read position in the message bytes
    cursor = 0
    payload_header_raw = message[cursor : cursor + _PAYLOAD_HEADER_SIZE]
    payload_header_record = np.frombuffer(
        payload_header_raw,
        dtype=payload_header_dtype,
        count=1,
    )[0]

    # validate payload header
    header_message_type = int(payload_header_record["message_type"])
    if header_message_type != _PAYLOAD_HEADER_TYPE:
        raise ValueError(
            "Binary payload header message_type mismatch: "
            f"expected {_PAYLOAD_HEADER_TYPE}, got {header_message_type}"
        )

    frame_number = int(payload_header_record["frame_number"])
    camera_count = int(payload_header_record["number_of_cameras"])
    if camera_count < 0:
        raise ValueError("Binary payload number_of_cameras cannot be negative")

    cursor += _PAYLOAD_HEADER_SIZE

    payload_footer_start = len(message) - _PAYLOAD_FOOTER_SIZE

    cameras: list[dict[str, Any]] = []
    for _ in range(camera_count):
        next_header_end = cursor + _FRAME_HEADER_SIZE
        if next_header_end > payload_footer_start:
            raise ValueError("Binary payload ended before frame headers were fully read")

        frame_header_raw = message[cursor:next_header_end]
        frame_header_record = np.frombuffer(frame_header_raw, dtype=frame_header_dtype, count=1)[0]
        frame_message_type = int(frame_header_record["message_type"])
        if frame_message_type != _FRAME_HEADER_TYPE:
            raise ValueError(
                "Binary frame header message_type mismatch: "
                f"expected {_FRAME_HEADER_TYPE}, got {frame_message_type}"
            )

        frame_header_frame_number = int(frame_header_record["frame_number"])
        if frame_header_frame_number != frame_number:
            raise ValueError(
                "Binary frame header frame_number mismatch: "
                f"expected {frame_number}, got {frame_header_frame_number}"
            )

        cursor = next_header_end

        jpeg_size = int(frame_header_record["jpeg_string_length"])
        if jpeg_size < 0:
            raise ValueError("Binary payload frame jpeg_string_length cannot be negative")

        next_jpeg_end = cursor + jpeg_size
        if next_jpeg_end > payload_footer_start:
            raise ValueError("Binary payload ended before JPEG bytes were fully read")

        jpeg_bytes = message[cursor:next_jpeg_end]
        cursor = next_jpeg_end

        camera_id_raw = bytes(frame_header_record["camera_id"])
        camera_id = camera_id_raw.split(b"\x00", 1)[0].decode("ascii", errors="strict")

        cameras.append(
            {
                "frame_header": {
                    "message_type": frame_message_type,
                    "frame_number": frame_header_frame_number,
                    "camera_id": camera_id,
                    "camera_index": int(frame_header_record["camera_index"]),
                    "image_width": int(frame_header_record["image_width"]),
                    "image_height": int(frame_header_record["image_height"]),
                    "color_channels": int(frame_header_record["color_channels"]),
                    "jpeg_string_length": jpeg_size,
                },
                "jpeg": jpeg_bytes,
            }
        )

    if cursor != payload_footer_start:
        raise ValueError("Binary payload has extra bytes before payload footer")

    payload_footer_raw = message[payload_footer_start:]
    payload_footer_record = np.frombuffer(
        payload_footer_raw,
        dtype=payload_footer_dtype,
        count=1,
    )[0]

    footer_message_type = int(payload_footer_record["message_type"])
    if footer_message_type != _PAYLOAD_FOOTER_TYPE:
        raise ValueError(
            "Binary payload footer message_type mismatch: "
            f"expected {_PAYLOAD_FOOTER_TYPE}, got {footer_message_type}"
        )

    footer_frame_number = int(payload_footer_record["frame_number"])
    if footer_frame_number != frame_number:
        raise ValueError(
            "Binary payload footer frame_number mismatch: "
            f"expected {frame_number}, got {footer_frame_number}"
        )

    footer_camera_count = int(payload_footer_record["number_of_cameras"])
    if footer_camera_count != camera_count:
        raise ValueError(
            "Binary payload footer number_of_cameras mismatch: "
            f"expected {camera_count}, got {footer_camera_count}"
        )

    timestamp_ns = time_ns()

    payload: dict[str, Any] = {
        "timestamp_ns": timestamp_ns,
        "frame_id": f"frame-{frame_number}",
        "payload": {
            "transport": "websocket-binary",
            "layout": {
                "endianness": "little",
                "alignment": "numpy.align=True",
                "payload_header_size": _PAYLOAD_HEADER_SIZE,
                "frame_header_size": _FRAME_HEADER_SIZE,
                "payload_footer_size": _PAYLOAD_FOOTER_SIZE,
            },
            "payload_header": {
                "message_type": header_message_type,
                "frame_number": frame_number,
                "number_of_cameras": camera_count,
            },
            "cameras": cameras, # jpeg bytes stored in each camera dict
            "payload_footer": {
                "message_type": footer_message_type,
                "frame_number": footer_frame_number,
                "number_of_cameras": footer_camera_count,
            },
        },
    }
    return payload


__all__ = ["SkellyApiBridgeAdapter"]