"""Tests for orchestration contract scaffolding."""

from architecture.core_interfaces import PipelinePlanner as CorePipelinePlanner
from architecture.core_interfaces import WorkflowBuilder as CoreWorkflowBuilder
from orchestration.planner import PipelinePlanner, WorkflowBuilder


def test_pipeline_planner_contract_is_canonical() -> None:
    """Ensure orchestration planner reuses the canonical core interface."""
    assert PipelinePlanner is CorePipelinePlanner


def test_workflow_builder_contract_is_canonical() -> None:
    """Ensure orchestration workflow builder reuses the canonical core interface."""
    assert WorkflowBuilder is CoreWorkflowBuilder
