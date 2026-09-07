"""
Regression tests for the ReAct loop's Act step.

The bug
-------
`ReActLoop._select_and_execute_action` accepted `available_actions` and never
looked at it. It returned, unconditionally::

    {"action": {"name": "continue", "params": {}},
     "observation": f"Completed reasoning step based on thought: {thought[:100]}",
     "state_updates": {"last_thought": thought}}

So no action was ever executed, and the "observation" was the agent's own
thought echoed back at it. A ReAct loop whose observations come from the agent
instead of the environment has lost the only thing that makes ReAct work: the
grounding. It cannot discover it was wrong, because nothing outside the model
ever speaks. The module is presented in the README and the architecture doc as
this project's ReAct implementation, so the pattern was being taught backwards.

These tests are self-contained (a fake LLM, fake actions) and need no API key.

Run: pytest tests/test_react_loop.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.reasoning import ReActLoop, ReasoningStepType


class _FakeResponse:
    def __init__(self, text):
        self.text = text


class _FakeLLM:
    """Returns queued replies; records the prompts it was asked."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return _FakeResponse(self._replies.pop(0) if self._replies else "thinking")


def check_content(state):
    """Look up the content record."""
    return {"content_checked": True, "toxicity": 0.91}


def escalate(state):
    """Send the item to a human reviewer."""
    return {"escalated": True}


def explode(state):
    """Always fails."""
    raise RuntimeError("upstream 503")


def test_the_chosen_action_is_actually_executed():
    llm = _FakeLLM(["thought one", '{"name": "check_content", "params": {}}'])
    loop = ReActLoop(llm, max_iterations=1)

    state = loop.run("moderate this", {"id": 1}, [check_content, escalate],
                     lambda s: s.get("content_checked", False))

    # The decisive assertion: the action's real return value reached the state.
    assert state["content_checked"] is True
    assert state["toxicity"] == 0.91


def test_the_observation_comes_from_the_action_not_from_the_thought():
    """The old code put `thought[:100]` here, which is the whole bug."""
    llm = _FakeLLM(["I should check the content first",
                    '{"name": "check_content", "params": {}}'])
    loop = ReActLoop(llm, max_iterations=1)
    loop.run("moderate this", {"id": 1}, [check_content], lambda s: False)

    observations = [s.content for s in loop.reasoning_history
                    if s.step_type is ReasoningStepType.OBSERVATION]
    assert observations
    assert "toxicity" in observations[0]
    assert "I should check the content first" not in observations[0]


def test_the_action_step_records_the_real_action_name():
    llm = _FakeLLM(["thought", '{"name": "escalate", "params": {}}'])
    loop = ReActLoop(llm, max_iterations=1)
    loop.run("task", {}, [check_content, escalate], lambda s: False)

    actions = [s.content for s in loop.reasoning_history
               if s.step_type is ReasoningStepType.ACTION]
    assert "escalate" in actions[0]
    assert "continue" not in actions[0]   # the old hardcoded name


def test_the_model_is_shown_the_actions_it_may_choose_from():
    llm = _FakeLLM(["thought", '{"name": "check_content", "params": {}}'])
    loop = ReActLoop(llm, max_iterations=1)
    loop.run("task", {}, [check_content, escalate], lambda s: False)

    selection_prompt = llm.prompts[-1]
    assert "check_content" in selection_prompt
    assert "escalate" in selection_prompt
    assert "Look up the content record." in selection_prompt   # the docstring


def test_a_failing_action_is_reported_as_an_observation_not_swallowed():
    """A failure the loop cannot see is a failure it will repeat."""
    llm = _FakeLLM(["thought", '{"name": "explode", "params": {}}'])
    loop = ReActLoop(llm, max_iterations=1)
    state = loop.run("task", {}, [explode], lambda s: False)

    observations = [s.content for s in loop.reasoning_history
                    if s.step_type is ReasoningStepType.OBSERVATION]
    assert "explode failed" in observations[0]
    assert "upstream 503" in observations[0]
    assert "last_error" in state


def test_an_invalid_choice_is_not_silently_replaced_with_a_default():
    llm = _FakeLLM(["thought", '{"name": "delete_everything", "params": {}}'])
    loop = ReActLoop(llm, max_iterations=1)
    loop.run("task", {}, [check_content], lambda s: False)

    observations = [s.content for s in loop.reasoning_history
                    if s.step_type is ReasoningStepType.OBSERVATION]
    assert "not an available action" in observations[0]
    assert "check_content" in observations[0]   # tells the loop what IS valid


def test_unparseable_model_output_is_surfaced():
    llm = _FakeLLM(["thought", "I think we should probably check it, maybe?"])
    loop = ReActLoop(llm, max_iterations=1)
    loop.run("task", {}, [check_content], lambda s: False)

    observations = [s.content for s in loop.reasoning_history
                    if s.step_type is ReasoningStepType.OBSERVATION]
    assert "Could not parse an action choice" in observations[0]


def test_an_empty_action_list_says_so_rather_than_pretending_to_act():
    llm = _FakeLLM(["thought"])
    loop = ReActLoop(llm, max_iterations=1)
    loop.run("task", {}, [], lambda s: False)

    observations = [s.content for s in loop.reasoning_history
                    if s.step_type is ReasoningStepType.OBSERVATION]
    assert "No actions are available" in observations[0]


def test_the_loop_stops_once_the_completion_check_passes():
    llm = _FakeLLM(["thought", '{"name": "check_content", "params": {}}'] * 5)
    loop = ReActLoop(llm, max_iterations=5)
    loop.run("task", {}, [check_content], lambda s: s.get("content_checked", False))

    actions = [s for s in loop.reasoning_history
               if s.step_type is ReasoningStepType.ACTION]
    assert len(actions) == 1, "should stop after the check succeeds"
