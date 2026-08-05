"""Probe Skelly websocket stream using the bridge adapter.

Usage example:
    python3 scripts/probe_skelly_ws_bridge.py \
      --uri ws://localhost:53117/skellycam/websocket \
      --source-id cam_front_rgb \
      --frames 5
"""

from __future__ import annotations

import argparse
from typing import Any

from adapters.skelly_api_bridge import SkellyApiBridgeAdapter
from schemas.capabilities import SensorType
from schemas.models import SourceDescriptor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Skelly websocket bridge adapter")
    parser.add_argument(
        "--uri",
        default="ws://localhost:53117/skellycam/websocket",
        help="Skelly websocket endpoint URI",
    )
    parser.add_argument(
        "--source-id",
        default="cam_front_rgb",
        help="Source id expected in decoded frames",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=3,
        help="Number of frames to read before exiting",
    )
    return parser.parse_args()


def _summarize_frame_payload(payload: dict[str, Any]) -> str:
    cameras = payload.get("cameras")
    if isinstance(cameras, list):
        return f"binary cameras={len(cameras)}"
    keys = ",".join(sorted(payload.keys())[:6])
    return f"json keys={keys}"


def main() -> int:
    args = parse_args()

    source = SourceDescriptor(
        source_id=args.source_id,
        source_type=SensorType.RGB_CAMERA,
        uri=args.uri,
    )
    adapter = SkellyApiBridgeAdapter(source)

    print(f"Connecting to {args.uri} as source_id={args.source_id}")
    adapter.connect()
    try:
        for idx in range(args.frames):
            frame = adapter.read()
            summary = _summarize_frame_payload(frame.payload)
            print(
                f"[{idx + 1}/{args.frames}] frame_id={frame.frame_id} "
                f"timestamp_ns={frame.timestamp_ns} {summary}"
            )
    finally:
        adapter.disconnect()
        print("Disconnected")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
