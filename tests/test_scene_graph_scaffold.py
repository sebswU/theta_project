"""Tests for scene graph structural contracts."""

from importlib import import_module

from architecture.scene_graph import (
    CameraNode,
    HumanNode,
    MeshNode,
    ObjectNode,
    RelationshipEdge,
    Scene,
    SensorNode,
)


def test_scene_can_be_constructed_with_defaults() -> None:
    """Scene defaults should expose empty typed entity collections."""
    scene = Scene()

    assert scene.humans == []
    assert scene.objects == []
    assert scene.meshes == []
    assert scene.cameras == []
    assert scene.sensors == []
    assert scene.relationships == []


def test_scene_collections_accept_typed_entities() -> None:
    """Typed consumers should append/read entities without dict assumptions."""
    scene = Scene()

    human = HumanNode(human_id="h-1")
    obj = ObjectNode(object_id="o-1", object_type="chair")
    mesh = MeshNode(mesh_id="m-1", vertex_count=3, face_count=1)
    camera = CameraNode(camera_id="c-1", width=640, height=480)
    sensor = SensorNode(sensor_id="s-1", sensor_type="rgb")
    edge = RelationshipEdge(source_id="h-1", target_id="o-1", relation_type="near")

    scene.humans.append(human)
    scene.objects.append(obj)
    scene.meshes.append(mesh)
    scene.cameras.append(camera)
    scene.sensors.append(sensor)
    scene.relationships.append(edge)

    assert scene.humans[0].human_id == "h-1"
    assert scene.objects[0].object_type == "chair"
    assert scene.meshes[0].face_count == 1
    assert scene.cameras[0].intrinsics == {}
    assert scene.sensors[0].sensor_type == "rgb"
    assert scene.relationships[0].relation_type == "near"


def test_scene_graph_exports_stable_shape() -> None:
    """Module exports should remain stable for scaffolded consumers."""
    expected_exports = {
        "CameraNode",
        "HumanNode",
        "MeshNode",
        "ObjectNode",
        "RelationshipEdge",
        "Scene",
        "SensorNode",
    }

    exported = set(import_module("architecture.scene_graph").__all__)
    assert exported == expected_exports
