"""Tests for declarative runtime configuration wiring."""

from __future__ import annotations

import json
from pathlib import Path

from orchestration import load_runtime_assembly
from schemas import Frame, FusionConfiguration, FusionRequest, InferenceRequest

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_config_files_parse_cleanly() -> None:
    """Config files should parse cleanly through the runtime loader."""
    assembly = load_runtime_assembly(CONFIG_DIR)

    assert [source.source_id for source in assembly.sources] == [
        "cam_front_rgb",
        "cam_left_depth",
        "imu_base",
    ]
    assert assembly.outputs == ["scene_graph", "web_dashboard"]


def test_referenced_components_exist() -> None:
    """Configured adapter and plugin symbols should import and register successfully."""
    assembly = load_runtime_assembly(CONFIG_DIR)

    assert assembly.model_registry.discover() == ["rtmpose", "vitpose"]
    assert assembly.plugin_registry.discover() == [
        "scene_graph",
        "temporal_fusion",
        "triangulation",
    ]


def test_config_changes_alter_wiring_as_expected(tmp_path: Path) -> None:
    """Changing pipeline allow-lists should change the assembled plan deterministically."""
    config_dir = _copy_config_dir(CONFIG_DIR, tmp_path)
    pipelines_path = config_dir / "pipelines.yaml"
    pipelines = json.loads(pipelines_path.read_text(encoding="utf-8"))
    pipelines["pipelines"]["realtime_multicam"]["models"] = ["rtmpose"]
    pipelines["pipelines"]["realtime_multicam"]["fusion"] = ["scene_graph"]
    pipelines_path.write_text(json.dumps(pipelines, indent=2), encoding="utf-8")

    assembly = load_runtime_assembly(config_dir)

    assert assembly.plan.selected_models == ["rtmpose"]
    assert assembly.plan.selected_plugins == ["scene_graph"]


def test_minimal_end_to_end_path_assembles_without_manual_edits() -> None:
    """Default config should assemble a deterministic plan and graph end to end."""
    assembly = load_runtime_assembly(CONFIG_DIR)

    assert assembly.pipeline_name == "realtime_multicam"
    assert assembly.plan.selected_models == ["rtmpose", "vitpose"]
    assert assembly.plan.selected_plugins == ["scene_graph", "temporal_fusion", "triangulation"]
    assert assembly.graph.workflow_id.endswith(assembly.plan.plan_id)
    assert any(node.node_type == "source" for node in assembly.graph.nodes)
    assert any(node.node_type == "model" for node in assembly.graph.nodes)
    assert any(node.node_type == "fusion_plugin" for node in assembly.graph.nodes)


def test_configured_runtime_smoke_path() -> None:
    """Configured runtime should support a minimal model and fusion smoke path."""
    assembly = load_runtime_assembly(CONFIG_DIR)
    source = assembly.sources[0]
    frame = Frame(frame_id="frame-1", timestamp_ns=1, source_id=source.source_id)

    model_name = assembly.plan.selected_models[0]
    model = assembly.model_registry.create(model_name)
    model.load()
    inference = model.infer(InferenceRequest(request_id="infer-1", frames=[frame]))

    plugin_name = assembly.plan.selected_plugins[0]
    plugin = assembly.plugin_registry.create(plugin_name)
    plugin.initialize(FusionConfiguration(plugin_name=plugin.__class__.__name__))
    fusion = plugin.process(FusionRequest(request_id="fuse-1", inputs=[frame]))

    assert inference.outputs["model_name"] == model_name
    assert inference.outputs["frame_count"] == 1
    assert fusion.outputs["frame_count"] == 1
    assert fusion.outputs["route"] == plugin.output_type()


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
