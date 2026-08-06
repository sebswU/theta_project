"""Tests for declarative runtime configuration wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adapters import SkellyApiBridgeAdapter, SkellyCamFrameAdapter, URISensorAdapter
from fusion.plugins import SceneGraphPlugin
from orchestration import load_runtime_assembly
from registry.cvpr_model import CVPRModel
from schemas import (
    Frame,
    FusionConfiguration,
    FusionRequest,
    FusionResponse,
    InferenceRequest,
    InferenceResponse,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_config_files_parse_cleanly() -> None:
    """Config files should parse cleanly through the runtime loader."""
    assembly = load_runtime_assembly(CONFIG_DIR)

    assert [source.source_id for source in assembly.sources] == [
        "cam_front_rgb",
        "cam_left_depth",
        "imu_base",
        "websocket_skellycam",
    ]
    assert assembly.outputs == ["scene_graph", "web_dashboard"]
    assert assembly.source_calibration_refs == {
        "cam_front_rgb": "configs/calibration/front_rgb.yaml",
        "websocket_skellycam": "configs/calibration/skelly_api_bridge.yaml",
    }


def test_referenced_components_exist() -> None:
    """Configured adapter and plugin symbols should import and register successfully."""
    assembly = load_runtime_assembly(CONFIG_DIR)

    assert assembly.model_registry.discover() == ["rtmpose", "vitpose"]
    assert assembly.plugin_registry.discover() == [
        "scene_graph",
        "temporal_fusion",
        "triangulation",
    ]
    assert assembly.source_adapter_classes["cam_front_rgb"] is SkellyCamFrameAdapter
    assert assembly.source_adapter_classes["cam_left_depth"] is SkellyCamFrameAdapter
    assert assembly.source_adapter_classes["imu_base"] is URISensorAdapter
    assert assembly.source_adapter_classes["websocket_skellycam"] is SkellyApiBridgeAdapter

    model_cls = assembly.model_registry.get("rtmpose")
    assert issubclass(model_cls, CVPRModel)
    plugin_cls = assembly.plugin_registry.get("scene_graph")
    assert plugin_cls is SceneGraphPlugin


def test_config_changes_alter_wiring_as_expected(tmp_path: Path) -> None:
    """Changing pipeline allow-lists should change the assembled plan deterministically."""
    config_dir = _copy_config_dir(CONFIG_DIR, tmp_path)
    pipelines_path = config_dir / "pipelines.yaml"
    pipelines = json.loads(pipelines_path.read_text(encoding="utf-8"))
    pipelines["pipelines"]["realtime_multicam"]["inputs"] = ["imu"]
    pipelines["pipelines"]["realtime_multicam"]["models"] = []
    pipelines["pipelines"]["realtime_multicam"]["fusion"] = []
    pipelines_path.write_text(json.dumps(pipelines, indent=2), encoding="utf-8")

    assembly = load_runtime_assembly(config_dir)

    assert [source.source_id for source in assembly.sources] == ["imu_base"]
    assert assembly.plan.selected_models == []
    assert assembly.plan.selected_plugins == []
    assert list(assembly.source_adapter_classes) == ["imu_base"]


def test_minimal_end_to_end_path_assembles_without_manual_edits() -> None:
    """SkellyCam minimal pipeline should assemble from config-only declarations."""
    assembly = load_runtime_assembly(CONFIG_DIR, pipeline_name="skellycam_minimal")

    assert assembly.pipeline_name == "skellycam_minimal"
    assert [source.source_id for source in assembly.sources] == [
        "cam_front_rgb",
        "websocket_skellycam",
    ]
    assert assembly.plan.selected_models == ["rtmpose"]
    assert assembly.plan.selected_plugins == ["scene_graph"]
    assert assembly.outputs == ["scene_graph"]
    assert assembly.source_adapter_classes["cam_front_rgb"] is SkellyCamFrameAdapter
    assert assembly.source_adapter_classes["websocket_skellycam"] is SkellyApiBridgeAdapter
    assert assembly.source_calibration_refs["cam_front_rgb"] == "configs/calibration/front_rgb.yaml"
    assert (
        assembly.source_calibration_refs["websocket_skellycam"]
        == "configs/calibration/skelly_api_bridge.yaml"
    )
    assert assembly.graph.workflow_id.endswith(assembly.plan.plan_id)
    assert any(node.node_type == "source" for node in assembly.graph.nodes)
    assert any(node.node_type == "model" for node in assembly.graph.nodes)
    assert any(node.node_type == "fusion_plugin" for node in assembly.graph.nodes)


def test_missing_adapter_symbol_fails_clearly(tmp_path: Path) -> None:
    """Assembly should fail with a clear error when adapter symbol is missing."""
    config_dir = _copy_config_dir(CONFIG_DIR, tmp_path)
    cameras_path = config_dir / "cameras.yaml"
    cameras = json.loads(cameras_path.read_text(encoding="utf-8"))
    missing_symbol = "adapters.skellycam_frame_adapter.MissingAdapter"
    cameras["sources"][0]["adapter"] = missing_symbol
    cameras_path.write_text(json.dumps(cameras, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match=f"Missing configured symbol: {missing_symbol}"):
        load_runtime_assembly(config_dir)


def test_missing_plugin_symbol_fails_clearly(tmp_path: Path) -> None:
    """Assembly should fail with a clear error when plugin symbol is missing."""
    config_dir = _copy_config_dir(CONFIG_DIR, tmp_path)
    plugins_path = config_dir / "plugins.yaml"
    plugins = json.loads(plugins_path.read_text(encoding="utf-8"))
    missing_symbol = "fusion.plugins.MissingPlugin"
    plugins["plugins"][0]["class"] = missing_symbol
    plugins_path.write_text(json.dumps(plugins, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match=f"Missing configured symbol: {missing_symbol}"):
        load_runtime_assembly(config_dir)


def test_missing_calibration_reference_fails_clearly(tmp_path: Path) -> None:
    """Assembly should fail fast with clear messaging for missing calibration artifacts."""
    config_dir = _copy_config_dir(CONFIG_DIR, tmp_path)
    cameras_path = config_dir / "cameras.yaml"
    cameras = json.loads(cameras_path.read_text(encoding="utf-8"))
    cameras["sources"][0]["calibration"] = "configs/calibration/missing_front_rgb.yaml"
    cameras_path.write_text(json.dumps(cameras, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="Referenced calibration file does not exist"):
        load_runtime_assembly(config_dir)


def test_skellycam_runtime_smoke_path() -> None:
    """SkellyCam minimal config should run a fast model + fusion smoke path."""
    assembly = load_runtime_assembly(CONFIG_DIR, pipeline_name="skellycam_minimal")
    source = assembly.sources[0]
    frame = Frame(frame_id="frame-1", timestamp_ns=1, source_id=source.source_id)

    assert assembly.pipeline_name == "skellycam_minimal"
    assert [source.source_id for source in assembly.sources] == ["cam_front_rgb", "websocket_skellycam"]
    assert assembly.plan.selected_models == ["rtmpose"]
    assert assembly.plan.selected_plugins == ["scene_graph"]

    model_name = assembly.plan.selected_models[0]
    model = assembly.model_registry.create(model_name)
    assert model.is_loaded is False
    model.load()
    assert model.is_loaded is True
    inference = model.infer(InferenceRequest(request_id="infer-1", frames=[frame]))
    assert isinstance(inference, InferenceResponse)

    plugin_name = assembly.plan.selected_plugins[0]
    plugin = assembly.plugin_registry.create(plugin_name)
    plugin.initialize(FusionConfiguration(plugin_name=plugin.__class__.__name__))
    fusion = plugin.process(FusionRequest(request_id="fuse-1", inputs=[frame]))
    assert isinstance(fusion, FusionResponse)

    assert inference.request_id == "infer-1"
    assert inference.outputs["model_name"] == model_name
    assert inference.outputs["frame_count"] == 1
    assert fusion.request_id == "fuse-1"
    assert fusion.outputs["frame_count"] == 1
    assert fusion.outputs["route"] == plugin.output_type()
    assert fusion.metadata["plugin_name"] == plugin.__class__.__name__
    assert fusion.metadata["output_type"] == plugin.output_type()


def _copy_config_dir(source_dir: Path, tmp_path: Path) -> Path:
    target_dir = tmp_path / "configs"
    target_dir.mkdir()
    calibration_dir = target_dir / "calibration"
    calibration_dir.mkdir()

    for path in source_dir.iterdir():
        if path.is_file():
            (target_dir / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    for path in (source_dir / "calibration").iterdir():
        (calibration_dir / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return target_dir
