"""The provider contract, checked against every registered adapter.

Only the stub can be exercised without a key, so the vendor adapters are
verified structurally: they exist, they expose ``build``, and their source obeys
the rules that make the seam hold. That is weaker than an integration test and
it catches the failures that actually happen when someone adds a fourth vendor -
a missing truncation check, a swallowed finish reason, a schema parameter that
was never wired up.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from src.config import RoleConfig
from src.providers import (
    Completion,
    Provider,
    TruncatedCompletion,
    Usage,
    get_provider,
    is_retryable,
    registered_providers,
)
from src.providers._common import block, read_block, read_text_block, text_block

ADAPTER_DIR = Path(__file__).resolve().parent.parent / "src" / "providers"
VENDOR_ADAPTERS = ["gemini", "openai_compatible", "anthropic"]


def stub_role(role: str = "judge") -> RoleConfig:
    return RoleConfig(
        role=role, provider="stub", model="stub", temperature=0.0,
        max_output_tokens=512, thinking_budget=None, api_key=None,
        base_url=None, extra={"seed": 1},
    )


class TestRegistry:
    def test_every_registered_adapter_imports_and_exposes_build(self):
        for name in registered_providers():
            module = importlib.import_module(f"src.providers.{name}")
            assert callable(module.build), f"{name} has no build()"

    def test_an_unknown_provider_names_the_registered_ones(self):
        role = RoleConfig(
            role="judge", provider="nope", model="", temperature=0.0,
            max_output_tokens=1, thinking_budget=None, api_key=None,
            base_url=None, extra={},
        )
        with pytest.raises(ValueError, match="stub"):
            get_provider(role)


class TestTheStubSatisfiesTheProtocol:
    def test_it_is_a_provider(self):
        assert isinstance(get_provider(stub_role()), Provider)

    def test_it_is_deterministic(self):
        """Non-deterministic offline behaviour would make every assertion in
        the suite flaky and every reported number unreproducible."""
        prompt = block("RECORD", {"id": "r", "subject": {"title": "T"}}) + "\n" + text_block(
            "ARTEFACT", "T is fine."
        ) + "\n" + text_block("RUBRIC", "[require: subject.title]")
        first = get_provider(stub_role()).complete(prompt).text
        second = get_provider(stub_role()).complete(prompt).text
        assert first == second

    def test_it_costs_nothing(self):
        assert get_provider(stub_role()).estimated_usd(10_000, 10_000) == 0.0

    def test_it_refuses_a_role_it_has_no_behaviour_for(self):
        provider = get_provider(stub_role())
        provider._cfg = RoleConfig(  # type: ignore[attr-defined]
            role="nonsense", provider="stub", model="stub", temperature=0.0,
            max_output_tokens=1, thinking_budget=None, api_key=None,
            base_url=None, extra={},
        )
        with pytest.raises(ValueError, match="nonsense"):
            provider.complete("x")

    def test_the_generator_produces_a_realistic_mix_not_a_happy_path(self):
        """A mock that always succeeds teaches that the pipeline works.

        The stub injects a defect chosen by hashing (record id, attempt), so a
        run exercises the pass path, the revise path and the drop path. If this
        ever comes back all-clean or all-broken, the offline demo has stopped
        being a demo of anything.
        """
        provider = get_provider(stub_role("generator"))
        outputs = {
            provider.complete(
                block("RECORD", {"id": f"r{i}", "subject": {"title": "Title", "attributes": ["a"]}})
                + "\n"
                + block("ATTEMPT", 0)
            ).text
            for i in range(12)
        }
        assert len(outputs) > 3, "the stub generator is not varying its output"


class TestPromptBlocks:
    def test_json_blocks_round_trip(self):
        payload = {"id": "x", "nested": {"k": [1, 2]}}
        assert read_block(block("RECORD", payload), "RECORD") == payload

    def test_text_blocks_round_trip(self):
        assert read_text_block(text_block("RUBRIC", "line one\nline two"), "RUBRIC") == (
            "line one\nline two"
        )

    def test_a_missing_block_is_none_not_an_exception(self):
        assert read_block("nothing here", "RECORD") is None

    def test_blocks_do_not_collide(self):
        prompt = block("RECORD", {"a": 1}) + "\n" + text_block("ARTEFACT", "text")
        assert read_block(prompt, "RECORD") == {"a": 1}
        assert read_text_block(prompt, "ARTEFACT") == "text"


class TestRetryPolicy:
    @pytest.mark.parametrize(
        "message",
        ["429 too many requests", "503 service unavailable", "deadline exceeded"],
    )
    def test_transient_http_failures_retry(self, message):
        assert is_retryable(Exception(message))

    @pytest.mark.parametrize(
        "message",
        ["httpx.RemoteProtocolError: Server disconnected", "connection reset by peer"],
    )
    def test_transport_failures_retry(self, message):
        """A retry policy written against status codes misses every failure
        below the status code, and those are the ones a long batch job hits. A
        118-call run dying on call 40 because a socket closed throws away 39
        completed calls for nothing."""
        assert is_retryable(Exception(message))

    def test_a_retired_model_id_fails_immediately(self):
        """Retrying a 400 four times makes the learner wait fifteen seconds for
        the same wrong answer, with the real message buried in retry noise."""
        assert not is_retryable(Exception("400 model not found: gemini-2.0-flash"))


class TestCompletion:
    def test_length_is_recognised_as_truncation(self):
        assert Completion(text="x", finish_reason="length").was_truncated
        assert not Completion(text="x", finish_reason="stop").was_truncated

    def test_truncation_has_its_own_exception_type(self):
        assert issubclass(TruncatedCompletion, RuntimeError)


class TestUsage:
    def test_cost_is_tallied_per_role(self):
        """The interesting question is not what the run cost but what the JUDGE
        cost, since it runs on every artefact plus once more per retry. That is
        what makes K a business decision rather than a hyperparameter."""
        usage = Usage()
        usage.add("judge", Completion(text="a", input_tokens=100, output_tokens=50), 0.01)
        usage.add("generator", Completion(text="b", input_tokens=10, output_tokens=5), 0.001)
        assert usage.by_role["judge"]["usd"] == pytest.approx(0.01)
        assert usage.total_calls == 2


class TestVendorAdaptersFollowTheRules:
    @pytest.mark.parametrize("name", VENDOR_ADAPTERS)
    def test_every_vendor_adapter_checks_for_truncation(self, name):
        """A truncated judge verdict does not look like a failure: the JSON
        prefix may still parse into a label, with a reason cut off after four
        words, and that fragment then steers the revision loop. Any adapter
        that skips this check reintroduces the bug for its vendor only, which
        is the hardest kind to find.
        """
        source = (ADAPTER_DIR / f"{name}.py").read_text(encoding="utf-8")
        assert "TruncatedCompletion" in source, f"{name} never raises on truncation"

    @pytest.mark.parametrize("name", VENDOR_ADAPTERS)
    def test_every_vendor_adapter_supports_structured_output(self, name):
        """"Reply in JSON" fails on roughly one call in ten. Across a tuning run
        that is the difference between a pipeline and a coin flip."""
        source = (ADAPTER_DIR / f"{name}.py").read_text(encoding="utf-8")
        assert "json_schema" in source

    @pytest.mark.parametrize("name", VENDOR_ADAPTERS)
    def test_vendor_sdks_are_imported_lazily(self, name):
        """A top-level import would force every user to install all three SDKs
        to run on one of them - or on none of them, offline."""
        tree = ast.parse((ADAPTER_DIR / f"{name}.py").read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                for imported in names:
                    assert imported.split(".")[0] not in {
                        "google", "openai", "anthropic"
                    }, f"{name} imports its SDK at module level"


class TestThinkingBudget:
    """The per-role reasoning budget, and the live failures that shaped it.

    Both findings below came from running against the real API, and neither is
    reachable offline - which is the argument for doing a live pass at all.
    """

    def _role(self, name: str, budget: int | None) -> RoleConfig:
        return RoleConfig(
            role=name, provider="gemini", model="gemini-3.5-flash",
            temperature=0.0, max_output_tokens=1024, thinking_budget=budget,
            api_key="x", base_url=None, extra={},
        )

    def _budget_for(self, role: RoleConfig, schema=None):
        from src.providers.gemini import GeminiProvider

        provider = GeminiProvider.__new__(GeminiProvider)  # no SDK client needed
        provider._cfg = role
        return GeminiProvider._thinking_budget_for(provider, schema)

    def test_an_explicit_budget_beats_the_role_heuristic(self):
        """The right budget is a property of the TASK, not of whether the call
        happens to carry a schema. The generator carries no schema and still
        wants thinking off - measured, it spent 1,746 thinking tokens to write a
        40-token sentence."""
        assert self._budget_for(self._role("generator", 0)) == 0

    def test_an_unset_budget_falls_back_to_the_heuristic(self):
        assert self._budget_for(self._role("judge", None)) == 0
        assert self._budget_for(self._role("reflector", None)) is None

    def test_a_schema_still_disables_thinking_when_nothing_is_configured(self):
        assert self._budget_for(self._role("reflector", None), {"type": "object"}) == 0

    def test_zero_is_not_confused_with_unset(self):
        """`if budget:` would read 0 as absent and silently leave thinking on -
        the same falsy-zero trap as the budget cap in src/runtime.py."""
        assert self._budget_for(self._role("reflector", 0)) == 0


class TestThinkingModeRejection:
    """LIVE FINDING: Pro models refuse thinking_budget=0.

    `400 INVALID_ARGUMENT: Budget 0 is invalid. This model only works in
    thinking mode.` Without the fallback, pointing the judge at a Pro model is a
    hard failure whose message gives no hint that the *thinking* setting is the
    cause rather than the schema or the key.
    """

    def test_the_specific_rejection_is_recognised(self):
        from src.providers.gemini import _rejects_zero_thinking

        assert _rejects_zero_thinking(
            Exception(
                "400 INVALID_ARGUMENT. {'error': {'message': 'Budget 0 is "
                "invalid. This model only works in thinking mode.'}}"
            )
        )

    def test_other_400s_are_not_swallowed(self):
        """Retrying an unrelated 400 without the thinking config produces the
        same failure more slowly, with the real cause buried under a warning
        about thinking."""
        from src.providers.gemini import _rejects_zero_thinking

        assert not _rejects_zero_thinking(Exception("400 model not found: gemini-9"))
        assert not _rejects_zero_thinking(Exception("401 invalid api key"))


class TestThinkingTokensAreCounted:
    def test_the_adapter_adds_thoughts_to_output_tokens(self):
        """LIVE FINDING: `thoughts_token_count` is reported separately from
        `candidates_token_count` and is billed as output.

        Measured on this project's generator prompt: 1,746 thinking tokens for a
        40-token sentence. A cost report built on the visible response alone was
        out by a factor of forty-four - in the direction that makes a retry
        budget look affordable.
        """
        source = (ADAPTER_DIR / "gemini.py").read_text(encoding="utf-8")
        assert "thoughts_token_count" in source, (
            "the Gemini adapter is not counting thinking tokens, so every cost "
            "figure it produces understates spend"
        )
