# tests/test_trajectory_store.py
"""Tests for TrajectoryStore (Phase 2.1)."""
import pytest
import tempfile
import os
from time import time, sleep

from vel.memory.trajectory_store import TrajectoryStore, Trajectory, ToolCallRecord


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        yield db_path


@pytest.fixture
def trajectory_store(temp_db):
    """Create a TrajectoryStore instance."""
    return TrajectoryStore(temp_db)


class TestTrajectoryStore:
    """Test suite for TrajectoryStore."""

    def test_start_and_finish_trajectory(self, trajectory_store):
        """Test basic trajectory recording."""
        run_id = "run-001"
        signature = {"intent": "planning", "domain": "api"}

        # Start trajectory
        traj_id = trajectory_store.start_trajectory(
            run_id=run_id,
            signature=signature,
            input_message="Help me design an API",
            agent_id="test-agent",
            session_id="session-001"
        )

        assert traj_id is not None
        assert traj_id > 0

        # Finish trajectory
        messages = [
            {"role": "user", "content": "Help me design an API"},
            {"role": "assistant", "content": "Here's the design..."}
        ]

        trajectory_store.finish_trajectory(
            run_id=run_id,
            messages=messages,
            final_answer="Here's the design...",
            step_count=2
        )

        # Retrieve and verify
        traj = trajectory_store.get_trajectory(run_id)
        assert traj is not None
        assert traj.run_id == run_id
        assert traj.agent_id == "test-agent"
        assert traj.session_id == "session-001"
        assert traj.input_message == "Help me design an API"
        assert traj.final_answer == "Here's the design..."
        assert traj.step_count == 2
        assert traj.signature == signature
        assert len(traj.messages) == 2
        assert traj.duration_ms is not None
        assert traj.duration_ms >= 0

    def test_record_tool_calls(self, trajectory_store):
        """Test tool call recording."""
        run_id = "run-002"

        trajectory_store.start_trajectory(
            run_id=run_id,
            signature={"intent": "research"},
            input_message="Search for info"
        )

        # Record tool calls
        trajectory_store.record_tool_call(
            run_id=run_id,
            step_index=0,
            tool_name="search",
            input={"query": "AI agents"},
            output={"results": ["result1", "result2"]},
            tool_call_id="tc-001",
            duration_ms=150
        )

        trajectory_store.record_tool_call(
            run_id=run_id,
            step_index=1,
            tool_name="summarize",
            input={"text": "..."},
            error="Timeout",
            duration_ms=5000
        )

        trajectory_store.finish_trajectory(
            run_id=run_id,
            messages=[{"role": "user", "content": "Search for info"}],
            final_answer="Summary..."
        )

        # Retrieve and verify tool calls
        traj = trajectory_store.get_trajectory(run_id)
        assert traj is not None
        assert len(traj.tool_calls) == 2

        tc0 = traj.tool_calls[0]
        assert tc0.tool_name == "search"
        assert tc0.input == {"query": "AI agents"}
        assert tc0.output == {"results": ["result1", "result2"]}
        assert tc0.error is None
        assert tc0.duration_ms == 150

        tc1 = traj.tool_calls[1]
        assert tc1.tool_name == "summarize"
        assert tc1.error == "Timeout"
        assert tc1.output is None

    def test_get_unevaluated_trajectories(self, trajectory_store):
        """Test retrieval of unevaluated trajectories."""
        # Create multiple trajectories
        for i in range(5):
            run_id = f"run-{i:03d}"
            trajectory_store.start_trajectory(
                run_id=run_id,
                signature={"intent": "test"},
                input_message=f"Test {i}"
            )
            trajectory_store.finish_trajectory(
                run_id=run_id,
                messages=[],
                final_answer=f"Answer {i}"
            )

        # All should be unevaluated
        unevaluated = trajectory_store.get_unevaluated_trajectories(limit=10)
        assert len(unevaluated) == 5

        # Mark some as evaluated
        trajectory_store.mark_evaluated(unevaluated[0].id, success=True, confidence=0.9)
        trajectory_store.mark_evaluated(unevaluated[1].id, success=False, confidence=0.7)

        # Only 3 should be unevaluated now
        unevaluated = trajectory_store.get_unevaluated_trajectories(limit=10)
        assert len(unevaluated) == 3

    def test_get_successful_unextracted(self, trajectory_store):
        """Test retrieval of successful, unextracted trajectories."""
        # Create trajectories
        for i in range(4):
            run_id = f"run-{i:03d}"
            trajectory_store.start_trajectory(
                run_id=run_id,
                signature={"intent": "test"},
                input_message=f"Test {i}"
            )
            trajectory_store.finish_trajectory(
                run_id=run_id,
                messages=[],
                final_answer=f"Answer {i}"
            )

        # Get all and mark evaluated
        trajs = trajectory_store.get_unevaluated_trajectories()

        # Mark 2 as successful, 2 as failed
        trajectory_store.mark_evaluated(trajs[0].id, success=True)
        trajectory_store.mark_evaluated(trajs[1].id, success=True)
        trajectory_store.mark_evaluated(trajs[2].id, success=False)
        trajectory_store.mark_evaluated(trajs[3].id, success=False)

        # Only 2 successful, unextracted
        successful = trajectory_store.get_successful_unextracted()
        assert len(successful) == 2

        # Mark one as extracted
        trajectory_store.mark_strategies_extracted(successful[0].id)

        # Only 1 remaining
        successful = trajectory_store.get_successful_unextracted()
        assert len(successful) == 1

    def test_query_trajectories(self, trajectory_store):
        """Test querying with filters."""
        # Create trajectories with different attributes
        trajectory_store.start_trajectory(
            run_id="run-a1",
            signature={"intent": "planning"},
            input_message="Plan",
            agent_id="agent-a",
            session_id="sess-1"
        )
        trajectory_store.finish_trajectory(run_id="run-a1", messages=[], final_answer="Done")

        trajectory_store.start_trajectory(
            run_id="run-a2",
            signature={"intent": "planning"},
            input_message="Plan 2",
            agent_id="agent-a",
            session_id="sess-2"
        )
        trajectory_store.finish_trajectory(run_id="run-a2", messages=[], final_answer="Done 2")

        trajectory_store.start_trajectory(
            run_id="run-b1",
            signature={"intent": "research"},
            input_message="Research",
            agent_id="agent-b",
            session_id="sess-1"
        )
        trajectory_store.finish_trajectory(run_id="run-b1", messages=[], final_answer="Done 3")

        # Query by agent_id
        agent_a_trajs = trajectory_store.get_trajectories(agent_id="agent-a")
        assert len(agent_a_trajs) == 2

        # Query by session_id
        sess_1_trajs = trajectory_store.get_trajectories(session_id="sess-1")
        assert len(sess_1_trajs) == 2

        # Query by signature
        planning_trajs = trajectory_store.get_trajectories(
            signature={"intent": "planning"}
        )
        assert len(planning_trajs) == 2

    def test_delete_trajectory(self, trajectory_store):
        """Test trajectory deletion."""
        run_id = "run-delete-me"

        trajectory_store.start_trajectory(
            run_id=run_id,
            signature={"intent": "test"},
            input_message="Delete me"
        )
        trajectory_store.record_tool_call(
            run_id=run_id,
            step_index=0,
            tool_name="test",
            input={}
        )
        trajectory_store.finish_trajectory(run_id=run_id, messages=[], final_answer="Bye")

        # Verify exists
        assert trajectory_store.get_trajectory(run_id) is not None

        # Delete
        result = trajectory_store.delete_trajectory(run_id)
        assert result is True

        # Verify deleted
        assert trajectory_store.get_trajectory(run_id) is None

        # Delete non-existent
        result = trajectory_store.delete_trajectory("non-existent")
        assert result is False

    def test_get_statistics(self, trajectory_store):
        """Test statistics gathering."""
        # Create and evaluate trajectories
        for i in range(10):
            run_id = f"run-{i:03d}"
            trajectory_store.start_trajectory(
                run_id=run_id,
                signature={"intent": "test"},
                input_message=f"Test {i}",
                agent_id="test-agent"
            )
            trajectory_store.finish_trajectory(
                run_id=run_id,
                messages=[],
                final_answer=f"Answer {i}"
            )

        trajs = trajectory_store.get_unevaluated_trajectories()

        # Evaluate some
        for i, traj in enumerate(trajs[:7]):
            trajectory_store.mark_evaluated(traj.id, success=(i < 5))

        stats = trajectory_store.get_statistics(agent_id="test-agent")

        assert stats["total"] == 10
        assert stats["evaluated"] == 7
        assert stats["successful"] == 5
        assert stats["failed"] == 2
        assert stats["pending_evaluation"] == 3
        assert stats["success_rate"] == 5 / 7

    def test_strategies_used_tracking(self, trajectory_store):
        """Test that strategies_used is properly stored."""
        run_id = "run-strategies"

        trajectory_store.start_trajectory(
            run_id=run_id,
            signature={"intent": "test"},
            input_message="Test"
        )

        trajectory_store.finish_trajectory(
            run_id=run_id,
            messages=[],
            final_answer="Done",
            strategies_used=[1, 2, 3]
        )

        traj = trajectory_store.get_trajectory(run_id)
        assert traj.strategies_used == [1, 2, 3]

    def test_error_tracking(self, trajectory_store):
        """Test that errors are properly stored."""
        run_id = "run-error"

        trajectory_store.start_trajectory(
            run_id=run_id,
            signature={"intent": "test"},
            input_message="Test"
        )

        trajectory_store.finish_trajectory(
            run_id=run_id,
            messages=[],
            error="Something went wrong"
        )

        traj = trajectory_store.get_trajectory(run_id)
        assert traj.error == "Something went wrong"
        assert traj.final_answer is None


class TestTrajectory:
    """Test Trajectory dataclass."""

    def test_from_row(self, trajectory_store):
        """Test Trajectory.from_row class method."""
        trajectory_store.start_trajectory(
            run_id="test-from-row",
            signature={"intent": "test"},
            input_message="Test",
            agent_id="agent",
            session_id="session"
        )
        trajectory_store.finish_trajectory(
            run_id="test-from-row",
            messages=[{"role": "user", "content": "Test"}],
            final_answer="Answer"
        )

        row = trajectory_store.db.execute(
            "SELECT * FROM rb_trajectories WHERE run_id = ?",
            ("test-from-row",)
        ).fetchone()

        traj = Trajectory.from_row(row)

        assert traj.run_id == "test-from-row"
        assert traj.agent_id == "agent"
        assert traj.session_id == "session"
        assert isinstance(traj.signature, dict)
        assert isinstance(traj.messages, list)
