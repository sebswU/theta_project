"""RTMPose ONNX runtime utilities for the registry adapter layer.

This module keeps ONNX inference and postprocessing isolated from registry
contracts so RTMPose integration stays a thin adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from time import perf_counter
from typing import Any

import numpy as np


@dataclass(slots=True)
class RTMPoseOnnxConfig:
    """Configuration for RTMPose ONNX inference."""

    onnx_path: str
    device: str = "cpu"
    input_size: tuple[int, int] = (192, 256)
    simcc_split_ratio: float = 2.0
    use_mmpose_decode: bool = False


class RTMPoseOnnxRuntime:
    """ONNXRuntime-backed RTMPose runner.

    The runner supports two postprocessing strategies:
    - mmpose decode utility (if available and enabled)
    - local fallback decode implementation
    """

    def __init__(self, config: RTMPoseOnnxConfig) -> None:
        self._config = config
        self._session: Any | None = None
        self._cv2: Any | None = None
        self._ort: Any | None = None
        self._mmpose_decoder: Any | None = None

    @property
    def input_size(self) -> tuple[int, int]:
        return self._config.input_size

    def load(self) -> None:
        """Load runtime dependencies and initialize ONNX session."""
        if self._session is not None:
            return

        try:
            cv2 = importlib.import_module("cv2")
        except ModuleNotFoundError as exc:
            raise RuntimeError("opencv-python is required for RTMPose ONNX adapter") from exc

        try:
            ort = importlib.import_module("onnxruntime")
        except ModuleNotFoundError as exc:
            raise RuntimeError("onnxruntime is required for RTMPose ONNX adapter") from exc

        providers = ["CPUExecutionProvider"]
        if self._config.device.lower() == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self._cv2 = cv2
        self._ort = ort
        self._session = ort.InferenceSession(self._config.onnx_path, providers=providers)

        if self._config.use_mmpose_decode:
            self._mmpose_decoder = _load_mmpose_decoder()

    def infer_image(self, image_bgr: np.ndarray) -> dict[str, Any]:
        """Run inference on a BGR image and return decoded keypoints."""
        if self._session is None or self._cv2 is None:
            raise RuntimeError("RTMPose runtime is not loaded. Call load() first.")

        preprocess_start = perf_counter()
        resized_img, center, scale = preprocess(image_bgr, self._config.input_size, self._cv2)
        preprocess_ms = (perf_counter() - preprocess_start) * 1000.0

        infer_start = perf_counter()
        outputs = _inference(self._session, resized_img)
        infer_ms = (perf_counter() - infer_start) * 1000.0

        post_start = perf_counter()
        keypoints, scores = postprocess(
            outputs,
            model_input_size=self._config.input_size,
            center=center,
            scale=scale,
            simcc_split_ratio=self._config.simcc_split_ratio,
            mmpose_decoder=self._mmpose_decoder,
        )
        post_ms = (perf_counter() - post_start) * 1000.0

        return {
            "keypoints": keypoints,
            "scores": scores,
            "timings_ms": {
                "preprocess": round(preprocess_ms, 3),
                "inference": round(infer_ms, 3),
                "postprocess": round(post_ms, 3),
            },
        }

    def decode_image_payload(self, payload: dict[str, Any]) -> np.ndarray:
        """Decode common frame payload formats into a BGR image."""
        if self._cv2 is None:
            raise RuntimeError("RTMPose runtime is not loaded. Call load() first.")

        image_value = payload.get("image")
        if isinstance(image_value, np.ndarray):
            if image_value.ndim != 3 or image_value.shape[2] != 3:
                raise ValueError("payload['image'] must be an HxWx3 numpy array")
            return image_value

        image_path = payload.get("image_path") or payload.get("path")
        if isinstance(image_path, str) and image_path.strip():
            img = self._cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Unable to read image from path: {image_path}")
            return img

        image_bytes = payload.get("image_bytes")
        if isinstance(image_bytes, (bytes, bytearray)):
            encoded = np.frombuffer(image_bytes, dtype=np.uint8)
            img = self._cv2.imdecode(encoded, self._cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Unable to decode payload['image_bytes']")
            return img

        raise ValueError(
            "Frame payload must include one of: image (numpy), image_path/path (str), "
            "or image_bytes (encoded image bytes)."
        )


def preprocess(
    img: np.ndarray,
    input_size: tuple[int, int],
    cv2: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess image for RTMPose ONNX inference."""
    img_shape = img.shape[:2]
    bbox = np.array([0, 0, img_shape[1], img_shape[0]], dtype=np.float32)
    center, scale = bbox_xyxy2cs(bbox, padding=1.25)
    resized_img, scale = top_down_affine(input_size, scale, center, img, cv2)

    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    resized_img = (resized_img.astype(np.float32) - mean) / std

    return resized_img, center, scale


def _inference(session: Any, img: np.ndarray) -> list[np.ndarray]:
    """Run ONNX inference for a single image."""
    model_input = img.transpose(2, 0, 1)[None, ...]
    session_input = {session.get_inputs()[0].name: model_input}
    session_output = [out.name for out in session.get_outputs()]
    outputs: list[np.ndarray] = session.run(session_output, session_input)
    return outputs


def postprocess(
    outputs: list[np.ndarray],
    model_input_size: tuple[int, int],
    center: np.ndarray,
    scale: np.ndarray,
    simcc_split_ratio: float,
    mmpose_decoder: Any | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode RTMPose outputs and map keypoints back to image space."""
    simcc_x, simcc_y = outputs

    if mmpose_decoder is not None:
        keypoints, scores = mmpose_decoder(simcc_x, simcc_y)
    else:
        keypoints, scores = decode(simcc_x, simcc_y, simcc_split_ratio)

    keypoints = keypoints / np.array(model_input_size, dtype=np.float32) * scale + center - scale / 2
    return keypoints, scores


def _load_mmpose_decoder() -> Any:
    """Load optional mmpose decode function."""
    try:
        codecs_utils = importlib.import_module("mmpose.codecs.utils")
        get_simcc_maximum = codecs_utils.get_simcc_maximum
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "use_mmpose_decode=True requires mmpose. Install mmpose or disable this option."
        ) from exc

    def _decode_with_mmpose(simcc_x: np.ndarray, simcc_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)
        return keypoints, scores

    return _decode_with_mmpose


def bbox_xyxy2cs(bbox: np.ndarray, padding: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Transform bbox format from xyxy to center/scale."""
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]

    return center, scale


def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float) -> np.ndarray:
    """Extend scale to match target aspect ratio."""
    w, h = np.hsplit(bbox_scale, [1])
    return np.where(
        w > h * aspect_ratio,
        np.hstack([w, w / aspect_ratio]),
        np.hstack([h * aspect_ratio, h]),
    )


def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a point by angle in radians."""
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]], dtype=np.float32)
    return rot_mat @ pt


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Get the third point needed for affine matrix calculation."""
    direction = a - b
    return b + np.r_[-direction[1], direction[0]]


def get_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: tuple[int, int],
    shift: tuple[float, float] = (0.0, 0.0),
    inv: bool = False,
    *,
    cv2: Any,
) -> np.ndarray:
    """Calculate affine transformation matrix for top-down pose input."""
    shift_np = np.array(shift, dtype=np.float32)
    src_w = scale[0]
    dst_w, dst_h = output_size

    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0.0, src_w * -0.5], dtype=np.float32), rot_rad)
    dst_dir = np.array([0.0, dst_w * -0.5], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift_np
    src[1, :] = center + src_dir + scale * shift_np
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def top_down_affine(
    input_size: tuple[int, int],
    bbox_scale: np.ndarray,
    bbox_center: np.ndarray,
    img: np.ndarray,
    cv2: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the bbox image as model input through affine transform."""
    w, h = input_size
    warp_size = (int(w), int(h))

    bbox_scale = _fix_aspect_ratio(bbox_scale, aspect_ratio=w / h)
    warp_mat = get_warp_matrix(bbox_center, bbox_scale, rot=0, output_size=(w, h), cv2=cv2)
    img_warped = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
    return img_warped, bbox_scale


def get_simcc_maximum(simcc_x: np.ndarray, simcc_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from SimCC representations."""
    n, k, _ = simcc_x.shape
    simcc_x_2d = simcc_x.reshape(n * k, -1)
    simcc_y_2d = simcc_y.reshape(n * k, -1)

    x_locs = np.argmax(simcc_x_2d, axis=1)
    y_locs = np.argmax(simcc_y_2d, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)

    max_val_x = np.amax(simcc_x_2d, axis=1)
    max_val_y = np.amax(simcc_y_2d, axis=1)
    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.0] = -1

    return locs.reshape(n, k, 2), vals.reshape(n, k)


def decode(
    simcc_x: np.ndarray,
    simcc_y: np.ndarray,
    simcc_split_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode model predictions from SimCC space to keypoint coordinates."""
    keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)
    keypoints = keypoints / simcc_split_ratio
    return keypoints, scores


__all__ = ["RTMPoseOnnxConfig", "RTMPoseOnnxRuntime"]