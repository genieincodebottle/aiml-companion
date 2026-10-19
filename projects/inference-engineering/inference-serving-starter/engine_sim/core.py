"""The engine_sim core: a virtual-time, step-by-step model of a continuous-batching engine.

No wall clock in here. Time is a float the engine advances itself, so the same
inputs always give the same outputs, byte for byte. The HTTP server (server.py)
and the offline incident generator (offline.py) both drive this class.

THE LATENCY MODEL (one engine step)
-----------------------------------
A step processes one token for every running decode sequence (k+1 tokens with
speculation) plus, in the same step, a chunk of prompt tokens (chunked prefill).

    memory_time  = (weight_bytes + kv_bytes_read) / (HBM bandwidth * bw_efficiency)
    compute_time = tokens_in_step * 2 * params     / (peak FLOPs     * mfu)
    step_time    = max(memory_time, compute_time) + step_overhead

* Decode at small batch is memory-bound: the weights are read once per step no
  matter how few sequences share the read. Quantisation shrinks weight_bytes, so
  it speeds decode up. Batching amortises the weight read, so throughput rises.
* Past the ridge point (compute_time > memory_time) every extra token costs real
  time. That is where speculation stops helping: verifying k+1 tokens per
  sequence multiplies compute_time while the accepted tokens do not.
* KV cache is finite. Admission needs free blocks; when decode growth runs out
  of blocks the youngest sequence is PREEMPTED (blocks freed, recomputed later).
* Prefix-cache hits skip prefill for the matching leading 16-token blocks.

SIMPLIFICATIONS (read these before you trust a number)
------------------------------------------------------
Attention FLOPs are ignored. KV cache is always 16-bit. Real kernels do not hit a
clean max(memory, compute). n-gram acceptance is a constant coin flip, not a
function of the text. Tensor parallel is one efficiency factor, with no
all-reduce model. The model is built to teach the SHAPE of the curves and the
metric signatures. It cannot give you a real tokens-per-second figure; for that
you run the gpu/ path.
"""
from __future__ import annotations

import heapq
import random
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Optional

from .config import BLOCK_SIZE, EngineConfig
from .metrics import Metrics


@dataclass
class Request:
    rid: str
    arrival: float
    prompt_ids: list
    max_tokens: int
    tag: str = ""


class TokenEvent(NamedTuple):
    rid: str
    n: int                 # tokens produced in this step for this request
    t: float               # virtual time at the end of the step
    first: bool            # True on the request's first output chunk
    done: bool
    arrival: float
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    error: Optional[str] = None


class Seq:
    __slots__ = ("req", "hashes", "computed", "generated", "target", "held", "n_shared", "pinned",
                 "admitted_t", "first_t", "last_t", "preemptions", "cached_tokens", "seen_admit")

    def __init__(self, req: Request):
        self.req = req
        self.hashes: list = []
        self.computed = 0      # tokens whose KV is stored (== prompt + generated once prefilled)
        self.generated = 0
        self.target = len(req.prompt_ids)   # tokens that must be prefilled before decode
        self.held = 0          # KV blocks held (shared + private)
        self.n_shared = 0      # leading blocks that are shared prefix-cache blocks
        self.pinned: list = []
        self.admitted_t: Optional[float] = None
        self.first_t: Optional[float] = None
        self.last_t: Optional[float] = None
        self.preemptions = 0
        self.cached_tokens = 0
        self.seen_admit = False

    @property
    def prefilled(self) -> bool:
        return self.computed >= self.target and self.held > 0


def _block_hashes(ids: list) -> list:
    out, prev = [], 0
    for i in range(len(ids) // BLOCK_SIZE):
        prev = hash((prev, tuple(ids[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE])))
        out.append(prev)
    return out


class Engine:
    def __init__(self, cfg: EngineConfig, accept_fn: Optional[Callable[[float], float]] = None):
        cfg.validate()
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.now = 0.0
        self.N = cfg.kv_capacity_blocks
        self.pending: list = []          # heap of (arrival, seqno, Request)
        self._seqno = 0
        self.waiting: deque = deque()
        self.running: list = []
        self.block_ref: dict = {}        # shared prefix blocks in use: hash -> refcount
        self.idle: OrderedDict = OrderedDict()   # cached blocks nobody uses (evictable, LRU)
        self.private_total = 0
        self.metrics = Metrics()
        self.accept_fn = accept_fn or (lambda t: cfg.spec_accept)
        self.k1 = cfg.spec_k + 1 if cfg.spec == "ngram" else 1
        self._bw = cfg.gpu_spec.hbm_gbps * 1e9 * cfg.tp_scale * cfg.bw_efficiency
        self._flops = cfg.peak_flops * cfg.tp_scale * cfg.mfu
        self._wbytes = cfg.weight_bytes
        self._kvb = cfg.kv_bytes_per_token
        self._2p = 2.0 * cfg.model_spec.params
        self.last_step: dict = {"batch": 0, "prefill_tokens": 0, "dt": 0.0, "mem_t": 0.0, "comp_t": 0.0}

    # ---- public API ---------------------------------------------------------
    def submit(self, req: Request) -> None:
        self._seqno += 1
        heapq.heappush(self.pending, (req.arrival, self._seqno, req))

    @property
    def busy(self) -> bool:
        return bool(self.running or self.waiting)

    def kv_usage(self) -> float:
        return (len(self.block_ref) + self.private_total) / self.N if self.N else 1.0

    def render_metrics(self, engine_id: str = "0") -> str:
        return self.metrics.render(self.cfg.model_spec.name, len(self.running), len(self.waiting),
                                   self.kv_usage(), engine_id)

    def reset_prefix_cache(self) -> None:
        self.idle.clear()

    def run_until(self, t: float) -> list:
        """Run every step whose START time is <= t. Returns the token events produced."""
        events: list = []
        while True:
            if not self.busy:
                if not self.pending or self.pending[0][0] > t:
                    break
            elif self.now > t:
                break
            events.extend(self.step())
        return events

    # ---- KV block accounting ------------------------------------------------
    def _free(self) -> int:
        return self.N - len(self.block_ref) - self.private_total - len(self.idle)

    def _ensure(self, need: int) -> bool:
        """Make `need` blocks free, evicting idle cached blocks LRU-first."""
        if need <= 0:
            return True
        if self.N - len(self.block_ref) - self.private_total < need:
            return False
        while self._free() < need and self.idle:
            self.idle.popitem(last=False)
        return self._free() >= need

    def _release(self, s: Seq) -> None:
        for h in s.pinned:
            self.block_ref[h] -= 1
            if self.block_ref[h] == 0:
                del self.block_ref[h]
                self.idle[h] = None
        self.private_total -= s.held - s.n_shared
        s.pinned, s.held, s.n_shared = [], 0, 0

    def _preempt(self, victim: Seq) -> None:
        self._release(victim)
        victim.computed = 0
        victim.target = len(victim.req.prompt_ids) + victim.generated   # recompute everything so far
        victim.preemptions += 1
        self.running.remove(victim)
        self.waiting.appendleft(victim)
        self.metrics.preemptions += 1

    def _try_admit(self, s: Seq) -> str:
        """Returns 'ok', 'wait' (no room yet) or 'reject' (can never fit)."""
        total_blocks = -(-s.target // BLOCK_SIZE)
        if total_blocks + 1 > self.N:
            return "reject"
        if not s.hashes:
            s.hashes = _block_hashes(s.req.prompt_ids)
        hits = 0
        if self.cfg.enable_prefix_caching:
            max_b = min(len(s.hashes), (s.target - 1) // BLOCK_SIZE)
            for h in s.hashes[:max_b]:
                if h in self.block_ref or h in self.idle:
                    hits += 1
                else:
                    break
        idle_hits = sum(1 for h in s.hashes[:hits] if h in self.idle and h not in self.block_ref)
        avail = self.N - len(self.block_ref) - self.private_total - idle_hits
        need = total_blocks - hits + 1
        if need > avail:
            return "wait"
        for h in s.hashes[:hits]:
            if h in self.block_ref:
                self.block_ref[h] += 1
            else:
                del self.idle[h]
                self.block_ref[h] = 1
            s.pinned.append(h)
        s.n_shared = hits
        s.held = total_blocks
        self.private_total += total_blocks - hits
        while self._free() < 0 and self.idle:
            self.idle.popitem(last=False)
        s.computed = hits * BLOCK_SIZE
        s.cached_tokens = hits * BLOCK_SIZE
        self.metrics.prefix_queries += len(s.req.prompt_ids)
        self.metrics.prefix_hits += hits * BLOCK_SIZE
        if not s.seen_admit:
            s.seen_admit = True
            self.metrics.queue.observe(max(0.0, self.now - s.req.arrival))
        s.admitted_t = self.now
        return "ok"

    def _promote(self, s: Seq) -> None:
        """After prefill, the prompt's full blocks become shareable prefix-cache blocks."""
        if not self.cfg.enable_prefix_caching:
            return
        for i in range(s.n_shared, len(s.hashes)):
            h = s.hashes[i]
            if h in self.block_ref:
                self.block_ref[h] += 1
            elif h in self.idle:
                del self.idle[h]
                self.block_ref[h] = 1
            else:
                self.block_ref[h] = 1
            # a brand-new block moves private -> shared (total unchanged); a block that
            # already existed means our private copy is a duplicate and is freed
            self.private_total -= 1
            s.pinned.append(h)
            s.n_shared += 1

    # ---- one engine step ----------------------------------------------------
    def step(self) -> list:
        while self.pending and self.pending[0][0] <= self.now:
            self.waiting.append(Seq(heapq.heappop(self.pending)[2]))
        if not self.running and not self.waiting:
            if not self.pending:
                return []
            self.now = max(self.now, self.pending[0][0])
            while self.pending and self.pending[0][0] <= self.now:
                self.waiting.append(Seq(heapq.heappop(self.pending)[2]))

        events: list = []
        k1 = self.k1

        # 1. decode growth, preempting the youngest sequence when the pool is full
        for s in list(self.running):
            if s not in self.running or not s.prefilled:
                continue
            need = -(-(s.computed + k1) // BLOCK_SIZE) - s.held
            if need <= 0:
                continue
            while not self._ensure(need):
                victim = self.running[-1]
                self._preempt(victim)
                if victim is s:
                    break
            else:
                s.held += need
                self.private_total += need

        dec = [s for s in self.running if s.prefilled]
        budget = max(0, self.cfg.max_num_batched_tokens - len(dec) * k1)

        # 2. prefill chunks: continue running prefills, then admit from the waiting queue
        chunks: list = []        # (seq, chunk_tokens, computed_before)
        for s in self.running:
            if not s.prefilled and budget > 0:
                c = min(s.target - s.computed, budget)
                chunks.append((s, c, s.computed))
                budget -= c
        while self.waiting and budget > 0 and len(self.running) < self.cfg.max_num_seqs:
            s = self.waiting[0]
            verdict = self._try_admit(s)
            if verdict == "reject":
                self.waiting.popleft()
                n_p = len(s.req.prompt_ids)
                events.append(TokenEvent(s.req.rid, 0, self.now, False, True, s.req.arrival, n_p, 0, 0,
                                         "prompt_too_long_for_kv_cache"))
                continue
            if verdict == "wait":
                break
            self.waiting.popleft()
            self.running.append(s)
            c = min(s.target - s.computed, budget)
            chunks.append((s, c, s.computed))
            budget -= c

        if not dec and not chunks:
            # nothing can run (cache full of pinned blocks): wait for the next arrival
            self.now = self.now + 0.001
            return events

        # 3. step time
        prefill_tokens = sum(c for _, c, _ in chunks)
        tokens = len(dec) * k1 + prefill_tokens
        kv_read = sum(s.computed for s in dec) + sum(before for _, _, before in chunks)
        mem_t = (self._wbytes + kv_read * self._kvb) / self._bw
        comp_t = tokens * self._2p / self._flops
        dt = max(mem_t, comp_t) + self.cfg.step_overhead_s
        if self.cfg.spec == "ngram":
            dt += self.cfg.spec_overhead_s
        t_end = self.now + dt
        self.last_step = {"batch": len(dec), "prefill_tokens": prefill_tokens, "dt": dt,
                          "mem_t": mem_t, "comp_t": comp_t}

        # 4. outputs
        m = self.metrics
        finished: list = []
        alpha = self.accept_fn(self.now) if self.cfg.spec == "ngram" else 0.0
        for s in dec:
            n_new = 1
            if self.cfg.spec == "ngram":
                acc = 0
                for _ in range(self.cfg.spec_k):
                    if self.rng.random() < alpha:
                        acc += 1
                    else:
                        break
                m.spec_drafts += 1
                m.spec_draft_tokens += self.cfg.spec_k
                m.spec_accepted += acc
                n_new = 1 + acc
            n_new = min(n_new, s.req.max_tokens - s.generated)
            s.computed += n_new
            s.generated += n_new
            m.generation_tokens += n_new
            m.itl.observe(t_end - s.last_t)
            s.last_t = t_end
            events.append(self._event(s, n_new, t_end, False))
            if s.generated >= s.req.max_tokens:
                finished.append(s)
        for s, c, _ in chunks:
            s.computed += c
            m.prompt_tokens += c
            if s.computed >= s.target:
                self._promote(s)
                s.computed += 1          # the sampled token joins the context
                first = s.first_t is None
                s.generated += 1
                m.generation_tokens += 1
                if first:
                    s.first_t = t_end
                    m.ttft.observe(t_end - s.req.arrival)
                else:
                    m.itl.observe(t_end - s.last_t)
                s.last_t = t_end
                events.append(self._event(s, 1, t_end, first))
                if s.generated >= s.req.max_tokens:
                    finished.append(s)
        for s in finished:
            m.e2e.observe(t_end - s.req.arrival)
            m.success += 1
            self._release(s)
            self.running.remove(s)
            events = [e._replace(done=True) if (e.rid == s.req.rid and e.t == t_end) else e for e in events]
        self.now = t_end
        return events

    def _event(self, s: Seq, n: int, t: float, first: bool) -> TokenEvent:
        return TokenEvent(s.req.rid, n, t, first, False, s.req.arrival, len(s.req.prompt_ids),
                          s.generated, s.cached_tokens)
