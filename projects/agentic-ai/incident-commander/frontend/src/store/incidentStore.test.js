import { describe, it, expect, beforeEach, vi } from 'vitest';

// The store imports the API client at module load, so it is mocked before import.
vi.mock('../services/api', () => ({
  scenarioAPI: { getScenarios: vi.fn() },
  incidentAPI: {
    triggerIncident: vi.fn(),
    listIncidents: vi.fn(),
    getIncident: vi.fn(),
    decideAction: vi.fn(),
    cancelIncident: vi.fn(),
    getUsage: vi.fn(),
    getConfig: vi.fn(),
  },
  memoryAPI: { getMemory: vi.fn(), search: vi.fn() },
  subscribeToIncident: vi.fn(() => () => {}),
}));

import { useIncidentStore, TERMINAL_STATUSES } from './incidentStore';
import { incidentAPI, scenarioAPI, subscribeToIncident } from '../services/api';

/**
 * The store is the SSE reducer: every loop event the backend emits lands here and
 * becomes what the operator sees. It was previously untested, which is how the
 * cost-accumulation bug below survived - the UI reported a number nobody checked
 * against the backend's.
 */

let seq = 0;
const evt = (type, payload = {}, extra = {}) => ({
  type,
  seq: ++seq,
  incident_id: 'inc_1',
  iteration: 1,
  payload,
  ...extra,
});

beforeEach(() => {
  seq = 0;
  useIncidentStore.getState().resetActive();
  vi.clearAllMocks();
});

describe('event ingestion', () => {
  it('appends events in order and tracks the iteration high-water mark', () => {
    const { pushEvent } = useIncidentStore.getState();
    pushEvent(evt('loop', {}, { iteration: 1 }));
    pushEvent(evt('loop', {}, { iteration: 2 }));
    pushEvent(evt('loop', {}, { iteration: 1 })); // a late/replayed event

    const state = useIncidentStore.getState();
    expect(state.events).toHaveLength(3);
    expect(state.iteration).toBe(2);
  });

  it('ignores malformed events instead of corrupting state', () => {
    const { pushEvent } = useIncidentStore.getState();
    pushEvent(null);
    pushEvent({});
    pushEvent({ notAType: true });
    expect(useIncidentStore.getState().events).toHaveLength(0);
  });

  it('deduplicates replayed events by seq', () => {
    // The backend replays history on reconnect, so a flaky connection would
    // otherwise render every step twice.
    const { pushEvent } = useIncidentStore.getState();
    const duplicate = evt('diagnostic', { leading_cause: 'x' });
    pushEvent(duplicate);
    pushEvent(duplicate);
    expect(useIncidentStore.getState().events).toHaveLength(1);
  });
});

describe('cost and usage accounting', () => {
  it('assigns cumulative totals rather than summing them', () => {
    // The backend sends the incident's RUNNING TOTAL on every token_usage event.
    // Adding them would multiply the displayed cost by the number of emits, and a
    // budget readout that overstates by 4x is worse than no readout at all.
    const { pushEvent } = useIncidentStore.getState();
    pushEvent(evt('token_usage', { tokens: 1000, cost_usd: 0.004 }));
    pushEvent(evt('token_usage', { tokens: 2500, cost_usd: 0.011 }));
    pushEvent(evt('token_usage', { tokens: 4000, cost_usd: 0.018 }));

    const state = useIncidentStore.getState();
    expect(state.totalTokens).toBe(4000);
    expect(state.totalCostUsd).toBeCloseTo(0.018);
  });

  it('tracks the budget and the per-agent breakdown', () => {
    useIncidentStore.getState().pushEvent(
      evt('token_usage', {
        tokens: 4000,
        cost_usd: 0.25,
        budget_usd: 0.5,
        budget_used_pct: 50,
        by_agent: { Diagnostician: { cost_usd: 0.2 }, Triage: { cost_usd: 0.05 } },
      })
    );
    const state = useIncidentStore.getState();
    expect(state.budgetUsd).toBe(0.5);
    expect(state.budgetUsedPct).toBe(50);
    expect(Object.keys(state.usageByAgent)).toEqual(['Diagnostician', 'Triage']);
  });
});

describe('the approval gate', () => {
  it('captures the pending action, including the critique', () => {
    useIncidentStore.getState().pushEvent(
      evt('awaiting_approval', {
        action_id: 'act_1',
        action_type: 'rollback_deploy',
        risk: 'high',
        critique: { endorse: true, strongest_objection: 'Assumes the deploy is the cause.' },
      })
    );
    const state = useIncidentStore.getState();
    expect(state.status).toBe('awaiting_approval');
    expect(state.pendingApproval.action_id).toBe('act_1');
    expect(state.pendingApproval.critique.strongest_objection).toBeTruthy();
  });

  it('clears the pending approval once the matching action executes', () => {
    const { pushEvent } = useIncidentStore.getState();
    pushEvent(evt('awaiting_approval', { action_id: 'act_1' }));
    pushEvent(evt('action_executed', { action_id: 'act_1', effective: true }));
    expect(useIncidentStore.getState().pendingApproval).toBeNull();
  });

  it('leaves a pending approval alone when a DIFFERENT action executes', () => {
    const { pushEvent } = useIncidentStore.getState();
    pushEvent(evt('awaiting_approval', { action_id: 'act_2' }));
    pushEvent(evt('action_executed', { action_id: 'act_1' }));
    expect(useIncidentStore.getState().pendingApproval.action_id).toBe('act_2');
  });

  it('does nothing when asked to decide with no pending approval', async () => {
    await useIncidentStore.getState().decideAction('approve');
    expect(incidentAPI.decideAction).not.toHaveBeenCalled();
  });

  it('ignores a second click while the first decision is still in flight', async () => {
    // The buttons disable on `approving`, but a fast double click lands before
    // React re-renders them - this guard is what actually prevents a double POST.
    useIncidentStore.setState({
      incidentId: 'inc_1',
      pendingApproval: { action_id: 'act_1' },
      approving: true,
    });
    await useIncidentStore.getState().decideAction('approve');
    expect(incidentAPI.decideAction).not.toHaveBeenCalled();
    useIncidentStore.setState({ approving: false }); // resetActive does not touch it
  });

  it('drops the gate and re-syncs on a 409 (action already decided elsewhere)', async () => {
    // Another tab decided first. Leaving the gate up would just invite repeated
    // 409s, so it clears, refreshes the snapshot, and keeps the explanation.
    incidentAPI.decideAction.mockRejectedValueOnce({
      response: { status: 409, data: { detail: 'action already decided' } },
    });
    incidentAPI.getIncident.mockResolvedValueOnce({
      incident_id: 'inc_1',
      status: 'executing',
      events: [],
    });
    useIncidentStore.setState({ incidentId: 'inc_1', pendingApproval: { action_id: 'act_1' } });
    await useIncidentStore.getState().decideAction('approve');

    const state = useIncidentStore.getState();
    expect(state.pendingApproval).toBeNull();
    expect(incidentAPI.getIncident).toHaveBeenCalledWith('inc_1');
    expect(state.error).toBe('action already decided');
  });

  it('surfaces a rejected approval as an error rather than failing silently', async () => {
    incidentAPI.decideAction.mockRejectedValueOnce({
      response: { data: { detail: 'action already resolved' } },
    });
    useIncidentStore.setState({ incidentId: 'inc_1', pendingApproval: { action_id: 'act_1' } });
    await useIncidentStore.getState().decideAction('approve');
    expect(useIncidentStore.getState().error).toBe('action already resolved');
  });
});

describe('stream reconnection', () => {
  it('re-subscribes to a live incident from the last seen seq', () => {
    // Leaving the console closes the stream; coming back must resume it, and
    // from the right place so the backend does not replay what we already have.
    const { pushEvent } = useIncidentStore.getState();
    useIncidentStore.setState({ incidentId: 'inc_1', status: 'diagnosing' });
    pushEvent(evt('loop'));
    pushEvent(evt('hypothesis', { hypotheses: [] }));

    useIncidentStore.getState().reconnectStream();
    expect(subscribeToIncident).toHaveBeenCalledWith(
      'inc_1',
      expect.any(Function),
      expect.objectContaining({ afterSeq: 2 })
    );
    useIncidentStore.getState().disconnectStream();
  });

  it('keeps the stream open after "resolved" until the postmortem and final status arrive', () => {
    const closeStream = vi.fn();
    subscribeToIncident.mockReturnValueOnce(closeStream);
    useIncidentStore.getState().connectStream('inc_1');
    const { pushEvent } = useIncidentStore.getState();
    pushEvent(evt('resolved', { reason: 'recovered' }));
    expect(closeStream).not.toHaveBeenCalled();
    pushEvent(evt('postmortem', { summary: 'fixed', actions_taken: [] }));
    expect(useIncidentStore.getState().postmortem).toMatchObject({ summary: 'fixed' });
    pushEvent(evt('status', { status: 'resolved', final: true }));
    expect(closeStream).toHaveBeenCalled();
  });

  it('reconnects to a resolved incident whose postmortem has not arrived yet', () => {
    useIncidentStore.setState({ incidentId: 'inc_1', status: 'resolved', postmortem: null });
    useIncidentStore.getState().reconnectStream();
    expect(subscribeToIncident).toHaveBeenCalled();
    useIncidentStore.getState().disconnectStream();
  });

  it('does not reconnect when the incident already finished', () => {
    useIncidentStore.setState({
      incidentId: 'inc_1',
      status: 'resolved',
      postmortem: { summary: 'done' },
    });
    useIncidentStore.getState().reconnectStream();
    expect(subscribeToIncident).not.toHaveBeenCalled();
  });

  it('does not reconnect when nothing has run yet', () => {
    useIncidentStore.getState().reconnectStream();
    expect(subscribeToIncident).not.toHaveBeenCalled();
  });
});

describe('terminal states', () => {
  it.each([
    ['resolved', 'recovered'],
    ['escalated', 'cost_ceiling'],
    ['cancelled', 'operator_cancelled'],
  ])('records %s with its halt reason and clears the gate', (type, reason) => {
    const { pushEvent } = useIncidentStore.getState();
    pushEvent(evt('awaiting_approval', { action_id: 'act_1' }));
    pushEvent(evt(type, { reason }));

    const state = useIncidentStore.getState();
    expect(state.status).toBe(type);
    expect(state.haltReason).toBe(reason);
    expect(state.pendingApproval).toBeNull();
  });

  it('treats cancelled as terminal', () => {
    // Missing from the original list, so a cancelled incident left the stream open.
    expect(TERMINAL_STATUSES).toContain('cancelled');
  });

  it('does not let an auto_approved status overwrite the loop status', () => {
    const { pushEvent } = useIncidentStore.getState();
    pushEvent(evt('status', { status: 'detected' }));
    pushEvent(evt('status', { status: 'auto_approved', action_id: 'act_1' }));
    expect(useIncidentStore.getState().status).toBe('detected');
  });
});

describe('agent reasoning events', () => {
  it('records recalled episodes from runbook memory', () => {
    useIncidentStore.getState().pushEvent(
      evt('memory_recall', {
        count: 1,
        episodes: [{ episode_id: 'inc_0', root_cause: 'heap exhaustion', score: 0.9 }],
      })
    );
    expect(useIncidentStore.getState().recalledEpisodes).toHaveLength(1);
  });

  it('tracks evidence-gathering rounds', () => {
    useIncidentStore
      .getState()
      .pushEvent(evt('evidence_request', { tools: ['get_config_diff'], round: 1, max_rounds: 2 }));
    expect(useIncidentStore.getState().evidenceRounds).toBe(1);
  });

  it('accumulates guardrail flags for the audit panel', () => {
    const { pushEvent } = useIncidentStore.getState();
    pushEvent(evt('guardrail', { kind: 'injection', message: 'quarantined' }));
    pushEvent(evt('guardrail', { kind: 'redaction', message: 'redacted a token' }));
    const flags = useIncidentStore.getState().guardrailFlags;
    expect(flags.map((f) => f.kind)).toEqual(['injection', 'redaction']);
  });

  it('records the supervisor route and whether an LLM chose it', () => {
    useIncidentStore
      .getState()
      .pushEvent(
        evt('route', { route: 'gather_evidence', reason: 'confidence too low', llm: true })
      );
    expect(useIncidentStore.getState().lastRoute).toEqual({
      route: 'gather_evidence',
      reason: 'confidence too low',
      llm: true,
    });
  });

  it('stores the postmortem payload', () => {
    useIncidentStore.getState().pushEvent(
      evt('postmortem', {
        summary: 'A deploy regressed checkout.',
        runbook_lesson: 'roll back first',
      })
    );
    expect(useIncidentStore.getState().postmortem.runbook_lesson).toBe('roll back first');
  });

  it('surfaces agent errors', () => {
    useIncidentStore.getState().pushEvent(evt('error', { kind: 'loop_crash', message: 'boom' }));
    expect(useIncidentStore.getState().error).toBe('boom');
  });
});

describe('lifecycle', () => {
  it('resets every field of the active incident', () => {
    const { pushEvent } = useIncidentStore.getState();
    pushEvent(evt('token_usage', { tokens: 100, cost_usd: 0.01 }));
    pushEvent(evt('guardrail', { kind: 'injection' }));
    pushEvent(evt('resolved', { reason: 'recovered' }));

    useIncidentStore.getState().resetActive();
    const state = useIncidentStore.getState();
    expect(state.events).toHaveLength(0);
    expect(state.totalCostUsd).toBe(0);
    expect(state.guardrailFlags).toHaveLength(0);
    expect(state.status).toBeNull();
    expect(state.haltReason).toBeNull();
    expect(state.postmortem).toBeNull();
  });

  it('hydrates from a stored incident so history opens fully populated', async () => {
    incidentAPI.getIncident.mockResolvedValueOnce({
      incident_id: 'inc_9',
      status: 'escalated',
      halt_reason: 'approval_timeout',
      engine: 'graph',
      events: [{ type: 'loop', seq: 1 }],
      iterations: 3,
      max_iterations: 6,
      evidence_rounds: 1,
      total_tokens: 5000,
      total_cost_usd: 0.02,
      max_cost_usd: 0.5,
      agent_usage: { Triage: { cost_usd: 0.001 } },
      guardrail_flags: [{ kind: 'injection' }],
      recalled_episodes: [],
      postmortem: { summary: 'x' },
    });

    await useIncidentStore.getState().loadIncident('inc_9');
    const state = useIncidentStore.getState();
    expect(state.status).toBe('escalated');
    expect(state.haltReason).toBe('approval_timeout');
    expect(state.engine).toBe('graph');
    expect(state.evidenceRounds).toBe(1);
    expect(state.guardrailFlags).toHaveLength(1);
  });

  it('reports a trigger failure instead of leaving the button spinning', async () => {
    incidentAPI.triggerIncident.mockRejectedValueOnce({
      response: { data: { detail: 'GOOGLE_API_KEY is not configured.' } },
    });
    const id = await useIncidentStore.getState().triggerIncident({ scenarioId: 'bad_deploy' });
    const state = useIncidentStore.getState();
    expect(id).toBeNull();
    expect(state.triggering).toBe(false);
    expect(state.error).toContain('GOOGLE_API_KEY');
  });

  it('loads the scenario catalog', async () => {
    scenarioAPI.getScenarios.mockResolvedValueOnce({
      scenarios: [{ id: 'bad_deploy', expected_outcome: 'resolved' }],
    });
    await useIncidentStore.getState().loadScenarios();
    expect(useIncidentStore.getState().scenarios).toHaveLength(1);
  });

  it('keeps working when the runtime config endpoint is unavailable', async () => {
    incidentAPI.getConfig.mockRejectedValueOnce(new Error('offline'));
    await useIncidentStore.getState().loadRuntimeConfig();
    expect(useIncidentStore.getState().runtimeConfig).toBeNull();
  });
});
