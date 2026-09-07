"""One call path for every role, with accounting and a hard budget stop.

Every model call in the project goes through :meth:`Runtime.call`. Providers are
built lazily and cached, usage is tallied per role, and spend is checked before
each call rather than after the run.

Why a budget stop is not optional
---------------------------------
An unbounded LLM pipeline's first failure signal is the invoice, and it arrives
a month late. This project has two loops that can run away: RART, which scores
the whole training split once per iteration, and the serving retry loop, which
multiplies every request by K. A bug in either is not a crash, it is a bill.

The cap is checked BEFORE the call. Checking afterwards means the call that
crosses the limit has already been paid for, which is the wrong side of the
line to notice you are over it.
"""

from __future__ import annotations

import logging
from typing import Any

from .config import Config
from .providers import Completion, Provider, Usage, get_provider

log = logging.getLogger(__name__)


class BudgetExceeded(RuntimeError):
    """The run was stopped because it reached its spend or call ceiling."""


class Runtime:
    def __init__(
        self,
        config: Config,
        *,
        max_usd: float | None = None,
        max_calls: int | None = None,
    ) -> None:
        self.config = config
        self.usage = Usage()
        self.max_usd = config.max_run_usd if max_usd is None else max_usd
        self.max_calls = max_calls
        self._providers: dict[str, Provider] = {}

    def provider(self, role: str) -> Provider:
        if role not in self._providers:
            self._providers[role] = get_provider(self.config.role(role))
        return self._providers[role]

    def call(
        self,
        role: str,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> Completion:
        self._check_budget(role)
        provider = self.provider(role)
        completion = provider.complete(
            prompt,
            system=system,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            json_schema=json_schema,
        )
        self.usage.add(
            role,
            completion,
            provider.estimated_usd(completion.input_tokens, completion.output_tokens),
        )
        return completion

    def _check_budget(self, role: str) -> None:
        if self.max_calls is not None and self.usage.total_calls >= self.max_calls:
            raise BudgetExceeded(
                f"call cap of {self.max_calls} reached before a {role} call. "
                f"Spend so far: ${self.usage.total_usd:.4f}. "
                "Raise the cap deliberately, or find out why the loop is not "
                "terminating - the second is usually the right question."
            )
        # `is not None`, not a truthiness check. A cap of 0.0 is falsy, so
        # `if self.max_usd` reads "spend nothing" and means "spend anything" -
        # the cap silently disables itself at exactly the value a cautious
        # person would set first. None is the way to say "no cap".
        if self.max_usd is not None and self.usage.total_usd >= self.max_usd:
            raise BudgetExceeded(
                f"estimated spend ${self.usage.total_usd:.4f} reached the cap of "
                f"${self.max_usd:.2f} before a {role} call. "
                "Raise MAX_RUN_USD in .env if this is expected. Per-role "
                f"breakdown: {self.usage.as_dict()['by_role']}"
            )

    # ------------------------------------------------------------- reporting
    def provenance(self) -> dict[str, Any]:
        """What produced these numbers. Stamped into every artefact the CLI writes.

        A results file that does not record which model produced it is a
        results file that will eventually be compared against another one that
        used a different model, and the difference will be attributed to
        whatever changed in the code. Two of the fields here exist purely to
        stop a later reader drawing a conclusion the run cannot support.
        """
        roles = {}
        for role in self.config.roles:
            rc = self.config.role(role)
            roles[role] = {"provider": rc.provider, "model": rc.model}
        return {
            "roles": roles,
            "usage": self.usage.as_dict(),
            # True when every role is the offline rule engine. These numbers
            # measure the rubric's rules, not any model's judgement.
            "offline_stub_run": self.config.is_fully_offline,
            # True when the generator and judge are the same model, so
            # self-preference bias is uncontrolled in whatever follows.
            "single_model_config": self.config.single_model_config,
        }
