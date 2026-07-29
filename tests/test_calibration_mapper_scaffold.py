"""Tests for SkellyCam calibration artifact mapping into canonical schemas."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchestration import CalibrationArtifactError, map_skellycam_calibration_artifact


def _write_artifact(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_calibration_mapper_parses_valid_artifact(tmp_path: Path) -> None:
    """Valid calibration artifact should map to canonical camera and calibration objects."""
    artifact = tmp_path / "front_rgb.yaml"
    _write_artifact(
        artifact,
        {
            "camera_id": "cam_front_rgb",
            "reference_frame": "rig/front_rgb",
            "intrinsics": {
                "fx": 721.0,
                "fy": 719.5,
                "cx": 640.0,
                "cy": 360.0,
                "width": 1280,
                "height": 720,
            },
            "extrinsics": [
                [1.0, 0.0, 0.0, 0.1],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.2],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
    )

    mapping = map_skellycam_calibration_artifact(artifact)

    assert mapping.camera_info.camera_id == "cam_front_rgb"
    assert mapping.camera_info.width == 1280
    assert mapping.camera_info.height == 720
    assert mapping.camera_info.intrinsics["fx"] == 721.0
    assert mapping.calibration.reference_frame == "rig/front_rgb"
    assert mapping.calibration.extrinsics[3][3] == 1.0


def test_calibration_mapper_rejects_missing_file(tmp_path: Path) -> None:
    """Missing calibration artifact should fail with explicit error."""
    artifact = tmp_path / "missing_calibration.yaml"

    with pytest.raises(CalibrationArtifactError, match="does not exist"):
        map_skellycam_calibration_artifact(artifact)


def test_calibration_mapper_rejects_bad_extrinsics_shape(tmp_path: Path) -> None:
    """Extrinsics must be a numeric 4x4 matrix."""
    artifact = tmp_path / "bad_extrinsics.yaml"
    _write_artifact(
        artifact,
        {
            "camera_id": "cam_front_rgb",
            "reference_frame": "rig/front_rgb",
            "intrinsics": {
                "fx": 721.0,
                "fy": 719.5,
                "cx": 640.0,
                "cy": 360.0,
                "width": 1280,
                "height": 720,
            },
            "extrinsics": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        },
    )

    with pytest.raises(CalibrationArtifactError, match="4x4 matrix"):
        map_skellycam_calibration_artifact(artifact)


def test_calibration_mapper_preserves_camera_identity(tmp_path: Path) -> None:
    """Runtime source identity should be preserved when source_id override is provided."""
    artifact = tmp_path / "identity.yaml"
    _write_artifact(
        artifact,
        {
            "camera_id": "artifact_camera_name",
            "reference_frame": "rig/front_rgb",
            "intrinsics": {
                "fx": 721.0,
                "fy": 719.5,
                "cx": 640.0,
                "cy": 360.0,
                "width": 1280,
                "height": 720,
            },
            "extrinsics": [
                [1.0, 0.0, 0.0, 0.1],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.2],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
    )

    mapping = map_skellycam_calibration_artifact(artifact, source_id="cam_front_rgb")

    assert mapping.camera_info.camera_id == "cam_front_rgb"
    assert mapping.calibration.calibration_id == "calib::cam_front_rgb"
