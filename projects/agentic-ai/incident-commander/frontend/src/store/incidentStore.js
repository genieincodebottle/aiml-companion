import { create } from 'zustand';
import { scenarioAPI, incidentAPI, subscribeToIncident } from '../services/api';

// Terminal statuses where the loop stops and the stream closes.
export const TERMINAL_STATUSES = ['resolved', 'escalated', 'cancelled', 'failed'];

// Event types that describe agent reasoning rather than loop control. The timeline
// renders these; the metrics panel ignores them.
export const REASONING_EVENTS = [
  'hypothesis',
  'diagnostic',
  'evidence_request',
  'evidence_gathered',
  'critique',
  'memory_recall',
];

let unsubscribeStream = null;

// Human-readable "what is the system doing right now", keyed by the event that
// most recently arrived. Without this the console goes silent for 10-30s during a
// model call and people conclude the trigger button did nothing and press it
// again - which was the single most common misreading of this UI.
export const ACTIVITY_BY_EVENT = {
  status: 'Setting up the incident...',
  loop: 'Starting a new iteration...',
  memory_recall: 'Searching runbook memory for similar past incidents...',
  route: 'Deciding which agent should act next...',
  hypothesis: 'Forming root-cause hypotheses...',
  diagnostic: 'Settling on a leading cause...',
  evidence_request: 'Not confident enough - requesting more telemetry...',
  evidence_gathered: 'Reading the new evidence...',
  proposed_action: 'Preparing the proposed fix...',
  critique: 'The Critic is reviewing the proposal...',
  awaiting_approval: 'Waiting for your approval. Nothing will run until you decide.',
  action_executed: 'Action applied. Watching for the effect...',
  verification: 'Checking fresh telemetry against the prediction...',
  postmortem: 'Writing the postmortem...',
  token_usage: null, // bookkeeping; never overwrite a meaningful message
};

// What an agent is doing while its model call is in flight (`step_start`).
export const ACTIVITY_BY_AGENT = {
  Triage: 'Triage is assessing severity and blast radius...',
  Diagnostician: 'The Diagnostician is analysing telemetry...',
  Remediation: 'Remediation is choosing one fix and predicting its effect...',
  Critic: 'The Critic is trying to find the flaw in that fix...',
  Verifier: 'The Verifier is judging whether the service actually recovered...',
  Postmortem: 'Postmortem is writing the report...',
  Supervisor: 'The Supervisor is routing...',
};

const initialActiveState = {
  incidentId: null,
  incident: null,
  status: null,
  //: When the trigger was accepted, for the elapsed-time readout.
  startedAt: null,
  //: One sentence describing what is happening right now.
  activity: null,
  //: True between a model call starting and its result arriving.
  agentBusy: false,
  haltReason: null,
  engine: null,
  events: [],
  streamConnected: false,
  pendingApproval: null,
  iteration: 0,
  maxIterations: null,
  evidenceRounds: 0,
  totalTokens: 0,
  totalCostUsd: 0,
  budgetUsd: null,
  budgetUsedPct: 0,
  usageByAgent: {},
  guardrailFlags: [],
  recalledEpisodes: [],
  lastRoute: null,
  postmortem: null,
  error: null,
};

export const useIncidentStore = create((set, get) => ({
  // Catalog
  scenarios: [],
  scenariosLoading: false,
  scenariosError: null,

  // Runtime config (the loop's actual ceilings, fetched from /api/config)
  runtimeConfig: null,

  // Active incident
  ...initialActiveState,
  triggering: false,
  approving: false,
  cancelling: false,

  // -------- Scenarios + config --------
  loadScenarios: async () => {
    set({ scenariosLoading: true, scenariosError: null });
    try {
      const data = await scenarioAPI.getScenarios();
      set({ scenarios: data.scenarios || [], scenariosLoading: false });
    } catch (err) {
      set({
        scenariosError: err?.message || 'Failed to load scenarios',
        scenariosLoading: false,
      });
    }
  },

  loadRuntimeConfig: async () => {
    try {
      set({ runtimeConfig: await incidentAPI.getConfig() });
    } catch {
      // Non-fatal: the console still works without the ceilings panel.
      set({ runtimeConfig: null });
    }
  },

  // -------- Trigger + stream --------
  triggerIncident: async ({ scenarioId, service, autoApprove }) => {
    set({ triggering: true, error: null });
    try {
      // Tear down any previous stream and reset live state.
      get().resetActive();
      const { incident_id: incidentId, engine } = await incidentAPI.triggerIncident({
        scenarioId,
        service,
        autoApprove,
      });
      set({
        incidentId,
        engine: engine || null,
        status: 'detected',
        triggering: false,
        startedAt: Date.now(),
        activity: 'Incident accepted. Connecting to the live stream...',
        agentBusy: true,
      });
      get().connectStream(incidentId);
      return incidentId;
    } catch (err) {
      set({
        error: err?.response?.data?.detail || err?.message || 'Failed to trigger incident',
        triggering: false,
      });
      return null;
    }
  },

  // Subscribe to an existing incident (e.g. opened from history) and replay.
  // `afterSeq` resumes the stream past events we already hold instead of
  // replaying the whole history.
  connectStream: (incidentId, { afterSeq = 0 } = {}) => {
    if (unsubscribeStream) {
      unsubscribeStream();
      unsubscribeStream = null;
    }
    set({ incidentId, streamConnected: false });
    unsubscribeStream = subscribeToIncident(incidentId, (evt) => get().pushEvent(evt), {
      afterSeq,
      onOpen: () => set({ streamConnected: true }),
      onError: () => set({ streamConnected: false }),
    });
  },

  // Re-attach to a still-running incident after the console unmounted. Leaving
  // the console closes the stream (disconnectStream below), and without this a
  // return visit showed the run frozen on "connecting" forever - the state was
  // all still here, but nothing ever re-subscribed. Resumes from the highest seq
  // we have seen so the backend streams only what we missed. No-op when idle,
  // already connected, or the incident has reached a terminal status.
  reconnectStream: () => {
    const { incidentId, status, streamConnected, postmortem } = get();
    if (!incidentId || streamConnected || unsubscribeStream) return;
    // A terminal run still has its postmortem to deliver; only a run that has it
    // is truly finished.
    if (!status || (TERMINAL_STATUSES.includes(status) && postmortem)) return;
    const afterSeq = get().events.reduce(
      (max, e) => (typeof e.seq === 'number' && e.seq > max ? e.seq : max),
      0
    );
    get().connectStream(incidentId, { afterSeq });
  },

  disconnectStream: () => {
    if (unsubscribeStream) {
      unsubscribeStream();
      unsubscribeStream = null;
    }
    set({ streamConnected: false });
  },

  // -------- Event ingestion --------
  pushEvent: (evt) => {
    if (!evt || !evt.type) return;
    const state = get();

    // The backend replays history on reconnect, and an SSE client can legitimately
    // see the same event twice. `seq` is monotonic per incident, so dropping
    // already-seen events here keeps the timeline from duplicating on a flaky
    // connection - which is the failure users actually notice.
    if (typeof evt.seq === 'number' && state.events.some((e) => e.seq === evt.seq)) {
      return;
    }

    const next = { events: [...state.events, evt] };
    const payload = evt.payload || {};

    if (typeof evt.iteration === 'number') {
      next.iteration = Math.max(state.iteration, evt.iteration);
    }

    // Keep a live "what is happening" line in sync with the stream. `step_start`
    // means a model call is in flight and is the only point where the UI would
    // otherwise appear frozen, so it drives the busy flag.
    if (evt.type === 'step_start') {
      next.agentBusy = true;
      next.activity = ACTIVITY_BY_AGENT[evt.agent] || `${evt.agent || 'An agent'} is working...`;
    } else if (evt.type !== 'token_usage') {
      next.agentBusy = false;
      const line = ACTIVITY_BY_EVENT[evt.type];
      if (line) next.activity = line;
    }

    switch (evt.type) {
      case 'status':
        if (payload.status && !String(payload.status).startsWith('auto_')) {
          next.status = payload.status;
        }
        if (payload.max_iterations != null) next.maxIterations = payload.max_iterations;
        if (payload.max_cost_usd != null) next.budgetUsd = payload.max_cost_usd;
        if (payload.engine) next.engine = payload.engine;
        if (payload.halt_reason) next.haltReason = payload.halt_reason;
        break;

      case 'route':
        next.lastRoute = { route: payload.route, reason: payload.reason, llm: payload.llm };
        break;

      case 'memory_recall':
        next.recalledEpisodes = payload.episodes || [];
        break;

      case 'evidence_request':
        if (payload.round != null) next.evidenceRounds = payload.round;
        break;

      case 'awaiting_approval':
        next.status = 'awaiting_approval';
        next.pendingApproval = payload;
        break;

      case 'action_executed':
        // Once an action executes, clear any matching pending approval.
        if (state.pendingApproval && payload.action_id === state.pendingApproval.action_id) {
          next.pendingApproval = null;
        }
        break;

      case 'guardrail':
        next.guardrailFlags = [...state.guardrailFlags, payload];
        break;

      case 'token_usage':
        // The backend sends CUMULATIVE totals for the incident, so these are
        // assigned, not accumulated. Summing them here would multiply the reported
        // cost by the number of emits - and a cost display that overstates by 4x is
        // worse than none, because people stop believing the budget.
        if (payload.tokens != null) next.totalTokens = payload.tokens;
        if (payload.cost_usd != null) next.totalCostUsd = payload.cost_usd;
        if (payload.budget_usd != null) next.budgetUsd = payload.budget_usd;
        if (payload.budget_used_pct != null) next.budgetUsedPct = payload.budget_used_pct;
        if (payload.by_agent) next.usageByAgent = payload.by_agent;
        break;

      case 'resolved':
        next.status = 'resolved';
        next.haltReason = payload.reason || 'recovered';
        next.pendingApproval = null;
        break;

      case 'escalated':
        next.status = 'escalated';
        next.haltReason = payload.reason || null;
        next.pendingApproval = null;
        break;

      case 'cancelled':
        next.status = 'cancelled';
        next.haltReason = payload.reason || 'operator_cancelled';
        next.pendingApproval = null;
        break;

      case 'postmortem':
        next.postmortem = payload;
        break;

      case 'error':
        next.error = payload.message || 'Agent reported an error';
        break;

      default:
        break;
    }

    if (next.status && TERMINAL_STATUSES.includes(next.status)) {
      next.agentBusy = false;
      next.activity = null;
    }

    set(next);

    // Close the stream on the backend's FINAL status event, not on the first
    // terminal one. The terminal event (resolved/escalated/...) is followed by the
    // postmortem and the final usage; closing early dropped the postmortem, so the
    // console never showed it.
    if (evt.type === 'status' && payload.final) {
      get().disconnectStream();
    }
  },

  // -------- Approval gate --------
  decideAction: async (decision, note) => {
    // Re-entrancy guard: the buttons disable on `approving`, but a fast double
    // click lands before React re-renders them, and would double-POST.
    if (get().approving) return;
    const { incidentId, pendingApproval } = get();
    if (!incidentId || !pendingApproval) return;
    set({ approving: true });
    try {
      await incidentAPI.decideAction(incidentId, {
        actionId: pendingApproval.action_id,
        decision,
        note,
      });
      // Clear locally, expect the stream to confirm via action_executed / status.
      set({ approving: false, pendingApproval: null });
    } catch (err) {
      const message =
        err?.response?.data?.detail || err?.message || 'Failed to submit approval decision';
      // 409 means the action was already decided differently - another tab, or a
      // race with auto-approve. Keeping the gate up would only invite more 409s,
      // so drop it and re-sync from the stored snapshot; the error text still
      // tells the operator why the gate vanished (set AFTER loadIncident, which
      // clears the error field).
      if (err?.response?.status === 409) {
        set({ approving: false, pendingApproval: null });
        await get().loadIncident(incidentId);
        set({ error: message });
        return;
      }
      set({ approving: false, error: message });
    }
  },

  cancelIncident: async () => {
    const { incidentId } = get();
    if (!incidentId) return;
    set({ cancelling: true });
    try {
      await incidentAPI.cancelIncident(incidentId);
      set({ cancelling: false });
    } catch (err) {
      set({ cancelling: false, error: err?.message || 'Failed to cancel incident' });
    }
  },

  // -------- Loading a stored incident (for history -> postmortem) --------
  loadIncident: async (incidentId) => {
    set({ error: null });
    try {
      const data = await incidentAPI.getIncident(incidentId);
      set({
        incident: data,
        incidentId: data.incident_id,
        status: data.status,
        haltReason: data.halt_reason || null,
        engine: data.engine || null,
        events: data.events || [],
        iteration: data.iterations || 0,
        maxIterations: data.max_iterations ?? null,
        evidenceRounds: data.evidence_rounds || 0,
        totalTokens: data.total_tokens || 0,
        totalCostUsd: data.total_cost_usd || 0,
        budgetUsd: data.max_cost_usd ?? null,
        usageByAgent: data.agent_usage || {},
        guardrailFlags: data.guardrail_flags || [],
        recalledEpisodes: data.recalled_episodes || [],
        postmortem: data.postmortem || null,
      });
      return data;
    } catch (err) {
      set({ error: err?.message || 'Failed to load incident' });
      return null;
    }
  },

  // -------- Reset --------
  resetActive: () => {
    if (unsubscribeStream) {
      unsubscribeStream();
      unsubscribeStream = null;
    }
    set({ ...initialActiveState });
  },
}));
