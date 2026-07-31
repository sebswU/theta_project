"""SkellyCam frame adapter that normalizes incoming payloads.

The adapter maps raw SkellyCam frame dictionaries into canonical
``schemas.models.Frame`` objects while enforcing required identity and
timestamp fields.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any

from adapters.sensor_adapter import SensorAdapter, SensorAdapterLifecycleError
from schemas.models import Frame


class SkellyCamFrameAdapter(SensorAdapter):
    """Adapter that wraps SkellyCam frames in the canonical Frame envelope."""

    def __init__(
        self,
        source,
        frames: (
            Iterable[Frame | dict[str, Any]]
            | Callable[[], Iterable[Frame | dict[str, Any]]]
            | None
        ) = None,
    ) -> None:
        super().__init__(source)
        if frames is None:
            frames = []
        self._frame_loader = frames if callable(frames) else None
        self._static_frames = None if callable(frames) else list(frames)
        self._frames_iter: Iterator[Frame | dict[str, Any]] | None = None

    def _connect(self) -> None:
        discovered = (
            self._frame_loader()
            if self._frame_loader is not None
            else self._static_frames or []
        )
        self._frames_iter = iter(discovered)

    def _read(self) -> Frame:
        if self._frames_iter is None:
            raise SensorAdapterLifecycleError("SkellyCam adapter is not connected")

        try:
            raw_frame = next(self._frames_iter)
        except StopIteration as exc:
            raise EOFError("No more SkellyCam frames available") from exc

        return _normalize_skellycam_frame(raw_frame, expected_source_id=self.source.source_id)

    def _disconnect(self) -> None:
        self._frames_iter = None


def _normalize_skellycam_frame(
    frame: Frame | dict[str, Any],
    *,
    expected_source_id: str,
) -> Frame:
    if isinstance(frame, Frame):
        if frame.source_id != expected_source_id:
            raise ValueError("SkellyCam frame source_id does not match adapter source")
        return frame

    if not isinstance(frame, dict):
        raise TypeError("Unsupported SkellyCam frame payload shape")

    source_id = frame.get("source_id")
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("SkellyCam frame is missing required source_id")
    source_id = source_id.strip()
    if source_id != expected_source_id:
        raise ValueError("SkellyCam frame source_id does not match adapter source")

    timestamp_ns = _parse_timestamp(frame)

    frame_id_raw = frame.get("frame_id") or frame.get("id")
    if isinstance(frame_id_raw, str) and frame_id_raw.strip():
        frame_id = frame_id_raw.strip()
    else:
        frame_id = f"{source_id}-{timestamp_ns}"

    payload = _build_payload(frame)
    return Frame(
        frame_id=frame_id,
        timestamp_ns=timestamp_ns,
        source_id=source_id,
        payload=payload,
    )


def _parse_timestamp(frame: dict[str, Any]) -> int:
    timestamp_value = frame.get("timestamp_ns", frame.get("timestamp"))
    if timestamp_value is None:
        raise ValueError("SkellyCam frame is missing required timestamp")
    if isinstance(timestamp_value, bool) or not isinstance(timestamp_value, (int, float)):
        raise ValueError("SkellyCam frame timestamp must be numeric")
    return int(timestamp_value)


def _build_payload(frame: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any]
    raw_payload = frame.get("payload")
    if raw_payload is None:
        payload = {}
    elif not isinstance(raw_payload, dict):
        raise ValueError("SkellyCam frame payload must be an object")
    else:
        payload = dict(raw_payload)

    passthrough = {
        key: value
        for key, value in frame.items()
        if key not in {"id", "frame_id", "source_id", "timestamp", "timestamp_ns", "payload"}
    }
    payload.update(passthrough)
    return payload


__all__ = ["SkellyCamFrameAdapter"]