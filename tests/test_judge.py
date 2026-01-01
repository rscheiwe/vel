# tests/test_judge.py
"""Tests for LLM-as-Judge (Phase 2.2)."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from vel.memory.judge import (
    LLMJudge,
    JudgeConfig,
    JudgeResult,
    JudgeOutcome,
    update_confidence_bayesian,
    DEFAULT_JUDGE_PROMPT,
)


class TestJudgeConfig:
    """Test JudgeConfig."""

    def test_defaults(self):
        """Test default configuration."""
        config = JudgeConfig()
        assert config.provider == "openai"
        assert config.model is None
        assert config.temperature == 0.0
        assert config.max_tokens == 1024
        assert config.timeout == 60.0
        assert config.batch_size == 5

    def test_get_model_with_explicit(self):
        """Test get_model with explicit model."""
        config = JudgeConfig(model="gpt-4")
        assert config.get_model() == "gpt-4"

    def test_get_model_provider_defaults(self):
        """Test get_model with provider defaults."""
        assert JudgeConfig(provider="openai").get_model() == "gpt-4o-mini"
        assert JudgeConfig(provider="anthropic").get_model() == "claude-3-haiku-20240307"
        assert JudgeConfig(provider="google").get_model() == "gemini-1.5-flash"

    def test_get_model_unknown_provider(self):
        """Test get_model with unknown provider falls back to gpt-4o-mini."""
        config = JudgeConfig(provider="unknown")
        assert config.get_model() == "gpt-4o-mini"


class TestJudgeResult:
    """Test JudgeResult."""

    def test_to_dict(self):
        """Test serialization to dict."""
        result = JudgeResult(
            trajectory_id="test-123",
            outcome=JudgeOutcome.SUCCESS,
            confidence=0.85,
            failure_notes=["note1"],
            reasoning="Good execution",
            usage={"total_tokens": 100},
            model="gpt-4o-mini",
            latency_ms=250.5,
            error=None
        )

        d = result.to_dict()

        assert d["trajectory_id"] == "test-123"
        assert d["outcome"] == "success"
        assert d["confidence"] == 0.85
        assert d["failure_notes"] == ["note1"]
        assert d["reasoning"] == "Good execution"
        assert d["usage"] == {"total_tokens": 100}
        assert d["model"] == "gpt-4o-mini"
        assert d["latency_ms"] == 250.5
        assert d["error"] is None


class TestLLMJudge:
    """Test LLMJudge."""

    @pytest.fixture
    def mock_llm_fn(self):
        """Create a mock LLM function."""
        async def fn(messages, model):
            return '{"outcome": "success", "confidence": 0.9, "reasoning": "Task completed", "failure_notes": []}'
        return fn

    @pytest.fixture
    def judge_with_mock(self, mock_llm_fn):
        """Create a judge with mock LLM."""
        return LLMJudge(config=JudgeConfig(), llm_fn=mock_llm_fn)

    def test_format_trajectory_with_dict(self, judge_with_mock):
        """Test formatting trajectory from dict."""
        traj = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            "tool_calls": []
        }

        formatted = judge_with_mock._format_trajectory(traj)
        assert "[Step 1] USER: Hello" in formatted
        assert "[Step 2] ASSISTANT: Hi there!" in formatted

    def test_format_trajectory_truncates_long_content(self, judge_with_mock):
        """Test that long content is truncated."""
        long_content = "x" * 1000
        traj = {
            "messages": [{"role": "user", "content": long_content}]
        }

        formatted = judge_with_mock._format_trajectory(traj)
        assert len(formatted) < 1000
        assert "..." in formatted

    def test_build_prompt(self, judge_with_mock):
        """Test prompt building."""
        traj = {
            "input_message": "Help me",
            "messages": [{"role": "user", "content": "Help me"}],
            "final_answer": "Done!"
        }

        prompt = judge_with_mock._build_prompt(traj)

        assert "Help me" in prompt
        assert "Done!" in prompt
        assert "<evaluation_task>" in prompt

    def test_parse_response_success(self, judge_with_mock):
        """Test parsing successful response."""
        response = '{"outcome": "success", "confidence": 0.85, "reasoning": "Good", "failure_notes": []}'

        result = judge_with_mock._parse_response(response, "test-id")

        assert result.outcome == JudgeOutcome.SUCCESS
        assert result.confidence == 0.85
        assert result.reasoning == "Good"
        assert result.failure_notes == []

    def test_parse_response_failure(self, judge_with_mock):
        """Test parsing failure response."""
        response = '{"outcome": "failure", "confidence": 0.7, "reasoning": "Bad", "failure_notes": ["mistake1"]}'

        result = judge_with_mock._parse_response(response, "test-id")

        assert result.outcome == JudgeOutcome.FAILURE
        assert result.confidence == 0.7
        assert result.failure_notes == ["mistake1"]

    def test_parse_response_with_markdown(self, judge_with_mock):
        """Test parsing response wrapped in markdown."""
        response = '''```json
{"outcome": "success", "confidence": 0.9, "reasoning": "OK", "failure_notes": []}
```'''

        result = judge_with_mock._parse_response(response, "test-id")
        assert result.outcome == JudgeOutcome.SUCCESS

    def test_parse_response_fallback(self, judge_with_mock):
        """Test fallback parsing for malformed response."""
        response = "This task was a complete success!"

        result = judge_with_mock._parse_response(response, "test-id")

        assert result.outcome == JudgeOutcome.SUCCESS
        assert result.confidence == 0.3  # Low confidence for fallback
        assert "Parse fallback" in result.reasoning

    def test_parse_response_failure_fallback(self, judge_with_mock):
        """Test fallback parsing detects failure."""
        response = "The agent failed to complete the task."

        result = judge_with_mock._parse_response(response, "test-id")
        assert result.outcome == JudgeOutcome.FAILURE

    @pytest.mark.asyncio
    async def test_evaluate(self, judge_with_mock):
        """Test full evaluation flow."""
        traj = {
            "run_id": "test-run",
            "input_message": "Do something",
            "messages": [{"role": "user", "content": "Do something"}],
            "final_answer": "Done"
        }

        result = await judge_with_mock.evaluate(traj)

        assert result.trajectory_id == "test-run"
        assert result.outcome == JudgeOutcome.SUCCESS
        assert result.confidence == 0.9
        assert result.model == "gpt-4o-mini"
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_evaluate_batch(self, judge_with_mock):
        """Test batch evaluation."""
        trajectories = [
            {"run_id": f"run-{i}", "input_message": "Test", "messages": [], "final_answer": "Done"}
            for i in range(3)
        ]

        results = await judge_with_mock.evaluate_batch(trajectories)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.trajectory_id == f"run-{i}"
            assert result.outcome == JudgeOutcome.SUCCESS

    @pytest.mark.asyncio
    async def test_evaluate_with_error(self):
        """Test evaluation handles errors gracefully."""
        async def failing_llm(messages, model):
            raise Exception("API Error")

        judge = LLMJudge(config=JudgeConfig(), llm_fn=failing_llm)

        result = await judge.evaluate({"run_id": "test"})

        assert result.outcome == JudgeOutcome.ERROR
        assert "API Error" in result.error


class TestUpdateConfidenceBayesian:
    """Test Bayesian confidence update function."""

    def test_success_increases_confidence(self):
        """Test that success multiplies confidence by 1.2."""
        new_conf = update_confidence_bayesian(0.5, success=True)
        assert new_conf == pytest.approx(0.6)  # 0.5 * 1.2

    def test_failure_decreases_confidence(self):
        """Test that failure multiplies confidence by 0.85."""
        new_conf = update_confidence_bayesian(0.5, success=False)
        assert new_conf == pytest.approx(0.425)  # 0.5 * 0.85

    def test_capped_at_max(self):
        """Test confidence is capped at max (0.95)."""
        new_conf = update_confidence_bayesian(0.9, success=True)
        assert new_conf == 0.95  # 0.9 * 1.2 = 1.08, capped to 0.95

    def test_floored_at_min(self):
        """Test confidence is floored at min (0.05)."""
        new_conf = update_confidence_bayesian(0.1, success=False)
        assert new_conf == pytest.approx(0.085)  # 0.1 * 0.85

        new_conf = update_confidence_bayesian(0.05, success=False)
        assert new_conf == 0.05  # Already at floor

    def test_custom_multipliers(self):
        """Test custom multipliers."""
        new_conf = update_confidence_bayesian(
            0.5,
            success=True,
            success_multiplier=1.5,
            failure_multiplier=0.5
        )
        assert new_conf == pytest.approx(0.75)

    def test_custom_bounds(self):
        """Test custom min/max bounds."""
        new_conf = update_confidence_bayesian(
            0.9,
            success=True,
            max_confidence=0.99
        )
        assert new_conf == 0.99

    def test_multiple_successes(self):
        """Test confidence growth over multiple successes."""
        conf = 0.5
        for _ in range(5):
            conf = update_confidence_bayesian(conf, success=True)

        # 0.5 * 1.2^5 = 1.24, capped at 0.95
        assert conf == 0.95

    def test_multiple_failures(self):
        """Test confidence decay over multiple failures."""
        conf = 0.5
        for _ in range(5):
            conf = update_confidence_bayesian(conf, success=False)

        # 0.5 * 0.85^5 ≈ 0.22
        assert conf == pytest.approx(0.5 * (0.85 ** 5))
