"""The Gateway itself - the request path every model call travels.

    redact -> route (tier->model) -> semantic cache -> provider (+fallback)
           -> account usage/cost -> audit

The gateway is also where two production guarantees live: a hard per-workflow cost
budget (checked before each call) and a complete audit record of every call. Because
all traffic funnels through here, these are platform-wide, not per-agent.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Type, TypeVar

from pydantic import BaseModel

from ..config import Settings
from ..models.schemas import GatewayUsage, ModelTier
from .cache import SemanticCache
from .providers import GeminiProvider, Provider, ProviderError, StubProvider
from .redaction import redact

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class GatewayError(RuntimeError):
    pass


class BudgetExceeded(GatewayError):
    """Raised when a call would push a workflow past its cost ceiling."""


class Gateway:
    """One gateway per platform process; shared by every tenant workflow."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cache = SemanticCache(threshold=settings.cache_similarity)
        self._audit_path = settings.data_dir / "audit"
        self._audit_path.mkdir(parents=True, exist_ok=True)
        # One provider per tier. Each tier's fallback is the other tier's model, so a
        # call is always answered by the model the audit record names.
        self._providers = self._build_providers(settings)
        # Per-workflow accumulated cost, enforced against settings.max_cost_usd. Cases
        # run on worker threads, so the ledger is locked.
        self._cost: dict[str, float] = {}
        self._cost_lock = threading.Lock()

    def _build_providers(self, s: Settings) -> dict[ModelTier, Provider]:
        if s.use_live_provider:  # pragma: no cover - needs key
            try:
                return {
                    ModelTier.PREMIUM: GeminiProvider(s.premium_model, s),
                    ModelTier.CHEAP: GeminiProvider(s.cheap_model, s),
                }
            except ProviderError as exc:
                logger.warning("Live provider init failed, using the offline provider: %s", exc)
        offline = StubProvider()
        return {ModelTier.PREMIUM: offline, ModelTier.CHEAP: offline}

    @property
    def provider_name(self) -> str:
        return self._providers[ModelTier.PREMIUM].name

    def _model_for(self, tier: ModelTier) -> str:
        if not self.settings.use_live_provider:
            return "offline"
        return self.settings.cheap_model if tier == ModelTier.CHEAP else self.settings.premium_model

    def cost_of(self, workflow_id: str) -> float:
        with self._cost_lock:
            return round(self._cost.get(workflow_id, 0.0), 6)

    def restore_cost(self, workflow_id: str, cost_usd: float) -> None:
        """Reload a resumed workflow's spend so its ceiling still counts what it already used."""
        with self._cost_lock:
            self._cost[workflow_id] = max(self._cost.get(workflow_id, 0.0), cost_usd)

    def complete(
        self,
        *,
        workflow_id: str,
        system: str,
        user: str,
        schema: Type[T],
        tier: ModelTier = ModelTier.PREMIUM,
        on_event=None,
    ) -> tuple[T, GatewayUsage]:
        """Route one structured call through the full gateway path.

        `on_event(name, payload)` is an optional sink so the runtime can fold gateway
        activity (cache hits, budget checks, the call itself) into the event log.
        """
        # 1. Budget check BEFORE spending anything.
        with self._cost_lock:
            spent = self._cost.get(workflow_id, 0.0)
        if spent >= self.settings.max_cost_usd:
            if on_event:
                on_event("budget_exceeded", {"spent": spent, "ceiling": self.settings.max_cost_usd})
            raise BudgetExceeded(f"workflow {workflow_id} hit cost ceiling ${self.settings.max_cost_usd}")

        # 2. PII redaction at the boundary.
        red_sys = redact(system) if self.settings.redact_pii else None
        red_usr = redact(user) if self.settings.redact_pii else None
        sys_text = red_sys.text if red_sys else system
        usr_text = red_usr.text if red_usr else user
        pii_masked = (red_sys.count if red_sys else 0) + (red_usr.count if red_usr else 0)

        # 3. Routing. The cache key is scoped to the workflow: two *cases* can look
        # semantically similar (templated prompts differ only in entity ids), and
        # cross-serving one case's screening/narrative to another would be a
        # correctness and audit failure, not a saving.
        model = self._model_for(tier)
        routing_key = f"{workflow_id}:{model}:{schema.__name__}"

        # 4. Semantic cache lookup.
        if self.settings.cache_enabled:
            cached = self.cache.lookup(usr_text, routing_key)
            if cached is not None:
                obj = schema.model_validate_json(cached)
                usage = GatewayUsage(model=model, cached=True, provider=self.provider_name)
                # Cache hits are still model-call decisions: audit them too, or the
                # audit log understates traffic served from cache.
                self._audit(workflow_id, model, schema.__name__, tier, usage, pii_masked)
                if on_event:
                    on_event("cache_hit", {"model": model, "schema": schema.__name__})
                return obj, usage

        # 5. Provider call, failing over to the other tier's model. The fallback is
        # guarded too: shared-cause failures (bad key, quota, outage) hit both, and an
        # unhandled ProviderError here would escape the gateway's own error contract.
        primary = self._providers[tier]
        fallback = self._providers[ModelTier.CHEAP if tier == ModelTier.PREMIUM else ModelTier.PREMIUM]
        try:
            obj, usage = primary.complete(sys_text, usr_text, schema)
        except ProviderError as exc:
            logger.warning("%s provider failed (%s); failing over.", tier.value, exc)
            if on_event:
                on_event("provider_failover", {"tier": tier.value, "error": str(exc)[:200]})
            try:
                obj, usage = fallback.complete(sys_text, usr_text, schema)
            except ProviderError as exc2:
                raise GatewayError(f"all providers failed: primary={exc}; fallback={exc2}") from exc2
            usage.model = f"{usage.model}(fallback)"
            model = usage.model

        # 6. Cost accounting.
        with self._cost_lock:
            self._cost[workflow_id] = self._cost.get(workflow_id, 0.0) + usage.cost_usd

        # 7. Cache store + audit.
        if self.settings.cache_enabled and not usage.cached:
            self.cache.store(usr_text, routing_key, obj.model_dump_json())
        self._audit(workflow_id, model, schema.__name__, tier, usage, pii_masked)
        if on_event:
            on_event(
                "gateway_call",
                {
                    "model": usage.model,
                    "tier": tier.value,
                    "schema": schema.__name__,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "cost_usd": round(usage.cost_usd, 6),
                    "pii_masked": pii_masked,
                },
            )
        return obj, usage

    def _audit(
        self,
        workflow_id: str,
        model: str,
        schema: str,
        tier: ModelTier,
        usage: GatewayUsage,
        pii_masked: int,
    ) -> None:
        """Append an immutable audit record. Every model call is accountable."""
        record = {
            "workflow_id": workflow_id,
            "model": model,
            "tier": tier.value,
            "schema": schema,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_usd": round(usage.cost_usd, 6),
            "pii_masked": pii_masked,
            "cached": usage.cached,
        }
        try:
            with (self._audit_path / "gateway.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError:  # pragma: no cover
            pass
