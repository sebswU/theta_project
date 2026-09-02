"""Interactive entry point for live pipeline demonstrations."""

from __future__ import annotations
import argparse
import signal
import socket
from collections.abc import Sequence
from urllib.parse import urlparse
from websocket import create_connection

from adapters import SensorAdapter
from orchestration.config_runtime import RuntimeAssembly, load_runtime_assembly

CONFIG_DIR = "configs"
PIPELINE_SOURCES = {
    "skellycam_rtmpose_open3d": ("websocket_skellycam",),
}
running = True

def stop_running(signum: int, frame: object) -> None:
    """Request a graceful shutdown after an interrupt or termination signal."""
    del signum, frame
    global running
    running = False

def websocket_is_available(uri: str, timeout_seconds: float = 2.0) -> bool:
    """Return whether the WebSocket endpoint's TCP listener is reachable."""
    parsed = urlparse(uri)
    if parsed.scheme not in {"ws", "wss"} or not parsed.hostname:
        return False
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)

    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False

def available_pipelines() -> list[str]:
    """Return demo pipelines whose required live endpoint is reachable."""
    available: list[str] = []
    for pipeline_name, source_ids in PIPELINE_SOURCES.items():
        assembly = load_runtime_assembly(CONFIG_DIR, pipeline_name=pipeline_name)
        sources_by_id = {source.source_id: source for source in assembly.sources}
        if all(
            source_id in sources_by_id
            and sources_by_id[source_id].uri is not None
            and websocket_is_available(str(sources_by_id[source_id].uri))
            for source_id in source_ids
        ):
            available.append(pipeline_name)
    return available

def choose_pipeline(candidates: Sequence[str]) -> str:
    """Prompt for one available pipeline."""
    if not candidates:
        raise RuntimeError(
            "No supported peripherals are reachable. Start SkellyCam at "
            "ws://localhost:53117/skellycam/websocket and try again."
        )
    print("Available pipelines:")
    for index, pipeline_name in enumerate(candidates, start=1):
        print(f"  {index}) {pipeline_name}")
    while True:
        selection = input(f"Select a pipeline [1-{len(candidates)}]: ").strip()
        if selection.isdigit() and 1 <= int(selection) <= len(candidates):
            return candidates[int(selection) - 1]
        print("Choose a listed pipeline number.")

def instantiate_adapters(assembly: RuntimeAssembly, pipeline_name: str) -> list[SensorAdapter]:
    """Instantiate only the verified live sources for the selected demo."""
    required_source_ids = set(PIPELINE_SOURCES[pipeline_name])
    return [
        assembly.source_adapter_classes[source.source_id](source)
        for source in assembly.sources
        if source.source_id in required_source_ids
    ]

def main() -> None:
    parser = argparse.ArgumentParser(description="Run a live Universal CV pipeline demo.")
    parser.add_argument("--pipeline", help="Run this available pipeline without prompting.")
    args = parser.parse_args()
    candidates = available_pipelines()
    if args.pipeline:
        if args.pipeline not in candidates:
            parser.error(f"Pipeline is unavailable: {args.pipeline}")
        pipeline_name = args.pipeline
    else:
        pipeline_name = choose_pipeline(candidates)

    assembly = load_runtime_assembly(CONFIG_DIR, pipeline_name=pipeline_name)
    adapters = instantiate_adapters(assembly, pipeline_name)
    signal.signal(signal.SIGINT, stop_running)
    signal.signal(signal.SIGTERM, stop_running)
    try:
        for adapter in adapters:
            adapter.connect()
        print(f"Running {pipeline_name}. Press Ctrl+C to stop.")
        while running:
            frames = [adapter.read() for adapter in adapters]
            print(f"Received {len(frames)} frame(s).")
    finally:
        for adapter in adapters:
            adapter.disconnect()

if __name__ == "__main__":
    main()

