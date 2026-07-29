"""Calibration artifact mapping for SkellyCam-compatible payloads.

This module imports declarative calibration artifacts and converts them into
canonical schema objects used by runtime wiring.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from schemas import Calibration, CameraInfo


class CalibrationArtifactError(ValueError):
    """Raised when a calibration artifact is missing or invalid."""


@dataclass(slots=True)
class CalibrationMapping:
    """Canonical calibration mapping for a single camera source."""

    camera_info: CameraInfo
    calibration: Calibration


def map_skellycam_calibration_artifact(
    artifact_path: str | Path,
    *,
    source_id: str | None = None,
) -> CalibrationMapping:
    """Map one SkellyCam calibration artifact to canonical schema objects.

    Args:
        artifact_path: Path to a JSON-compatible YAML calibration payload.
        source_id: Optional source identity override to preserve runtime source IDs.

    Returns:
        Canonical camera and calibration objects for runtime assembly.

    Raises:
        CalibrationArtifactError: If file is missing or payload shape is invalid.
    """
    path = Path(artifact_path)
    if not path.exists():
        raise CalibrationArtifactError(
            f"Calibration artifact file does not exist: {path}"
        )

    payload = _load_payload(path)

    camera_id_raw = source_id or payload.get("camera_id")
    if not isinstance(camera_id_raw, str) or not camera_id_raw.strip():
        raise CalibrationArtifactError("Calibration artifact must include non-empty camera_id")
    camera_id = camera_id_raw.strip()

    intrinsics = _parse_intrinsics(payload)
    extrinsics = _parse_extrinsics(payload)

    reference_frame_raw = payload.get("reference_frame", f"rig/{camera_id}")
    if not isinstance(reference_frame_raw, str) or not reference_frame_raw.strip():
        raise CalibrationArtifactError("reference_frame must be a non-empty string")
    reference_frame = reference_frame_raw.strip()

    calibration_id_raw = payload.get("calibration_id", f"calib::{camera_id}")
    if not isinstance(calibration_id_raw, str) or not calibration_id_raw.strip():
        raise CalibrationArtifactError("calibration_id must be a non-empty string")
    calibration_id = calibration_id_raw.strip()

    camera_info = CameraInfo(
        camera_id=camera_id,
        width=intrinsics["width"],
        height=intrinsics["height"],
        intrinsics={
            "fx": intrinsics["fx"],
            "fy": intrinsics["fy"],
            "cx": intrinsics["cx"],
            "cy": intrinsics["cy"],
        },
    )
    calibration = Calibration(
        calibration_id=calibration_id,
        reference_frame=reference_frame,
        extrinsics=extrinsics,
    )

    return CalibrationMapping(camera_info=camera_info, calibration=calibration)


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise CalibrationArtifactError(
            f"Calibration artifact is not valid JSON/YAML object: {path}"
        ) from exc

    if not isinstance(payload, dict):
        raise CalibrationArtifactError("Calibration artifact root must be an object")
    return payload


def _parse_intrinsics(payload: dict[str, Any]) -> dict[str, float | int]:
    raw_intrinsics = payload.get("intrinsics")
    if not isinstance(raw_intrinsics, dict):
        raise CalibrationArtifactError("intrinsics must be an object")

    values: dict[str, float | int] = {}
    for key in ("fx", "fy", "cx", "cy"):
        raw_value = raw_intrinsics.get(key)
        if not isinstance(raw_value, (int, float)):
            raise CalibrationArtifactError(f"intrinsics.{key} must be numeric")
        values[key] = float(raw_value)

    width = raw_intrinsics.get("width", payload.get("width"))
    height = raw_intrinsics.get("height", payload.get("height"))

    if isinstance(raw_intrinsics.get("resolution"), list) and len(raw_intrinsics["resolution"]) == 2:
        width = raw_intrinsics["resolution"][0]
        height = raw_intrinsics["resolution"][1]

    if not isinstance(width, int) or width <= 0:
        raise CalibrationArtifactError("intrinsics width must be a positive integer")
    if not isinstance(height, int) or height <= 0:
        raise CalibrationArtifactError("intrinsics height must be a positive integer")

    values["width"] = width
    values["height"] = height
    return values


def _parse_extrinsics(payload: dict[str, Any]) -> list[list[float]]:
    raw_extrinsics = payload.get("extrinsics")
    if not isinstance(raw_extrinsics, list):
        raise CalibrationArtifactError("extrinsics must be a 4x4 matrix")
    if len(raw_extrinsics) != 4:
        raise CalibrationArtifactError("extrinsics must be a 4x4 matrix")

    matrix: list[list[float]] = []
    for row in raw_extrinsics:
        if not isinstance(row, list) or len(row) != 4:
            raise CalibrationArtifactError("extrinsics must be a 4x4 matrix")
        parsed_row: list[float] = []
        for value in row:
            if not isinstance(value, (int, float)):
                raise CalibrationArtifactError("extrinsics entries must be numeric")
            parsed_row.append(float(value))
        matrix.append(parsed_row)
    return matrix


__all__ = [
    "CalibrationArtifactError",
    "CalibrationMapping",
    "map_skellycam_calibration_artifact",
]
