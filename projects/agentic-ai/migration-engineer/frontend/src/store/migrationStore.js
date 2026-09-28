import { create } from 'zustand';
import { systemAPI, jobAPI, runAPI, subscribeToRun } from '../services/api';

// Job statuses where the run is over and the stream can be closed.
export const TERMINAL_JOB_STATUSES = ['done', 'failed'];

let unsubscribeStream = null;

// ---------------------------------------------------------------------------
// Pure derived-state helpers. The whole fleet board is a projection of the
// ordered event stream, so we keep a single reducer that both the live stream
// (pushEvent) and history replay (loadRun) run through.
// ---------------------------------------------------------------------------

function makeRepo(repoId, name) {
  return {
    repo_id: repoId,
    name: name || repoId,
    status: 'queued',
    steps: 0,
    tool_calls: 0,
    files_changed: [],
    tokens: 0,
    cost_usd: 0,
    tests_passing: null,
    review_verdict: null,
    diff: null,
    summary: null,
    pr: null,
    guardrail_blocks: [],
    escalation_reason: null,
    activity: [],
  };
}

// The slice of state that is a pure projection of the event stream.
function emptyDerived() {
  return {
    title: null,
    mode: null,
    plan: null,
    jobStatus: null,
    repos: {},
    reposOrder: [],
    events: [],
    pendingApprovals: {},
    summary: null,
    totalTokens: 0,
    totalCostUsd: 0,
  };
}

const doneTone = (status) => {
  if (status === 'pr_open') return 'success';
  if (status === 'rejected' || status === 'failed') return 'error';
  return 'warning';
};

/**
 * Reduce a single MigrationEvent against the current derived state.
 * Returns a partial patch (safe to spread into zustand set or a fold).
 */
export function reduceEvent(state, evt) {
  const patch = { events: [...state.events, evt] };
  const p = evt.payload || {};
  const ts = evt.ts;

  // Immutably create/update the repo lane keyed by repoId.
  const withRepo = (id, name, mutate) => {
    if (!id) return;
    const repos = patch.repos || { ...state.repos };
    let order = patch.reposOrder || state.reposOrder;
    let repo = repos[id];
    if (!repo) {
      repo = makeRepo(id, name);
      order = [...order, id];
    } else {
      repo = { ...repo };
      if (name && repo.name === repo.repo_id) repo.name = name;
    }
    mutate(repo);
    repos[id] = repo;
    patch.repos = repos;
    patch.reposOrder = order;
  };

  const pushActivity = (repo, entry) => {
    repo.activity = [...repo.activity, { ts, ...entry }];
  };

  switch (evt.type) {
    case 'job_started':
      patch.jobStatus = 'running';
      if (p.title) patch.title = p.title;
      if (p.mode) patch.mode = p.mode;
      break;

    case 'plan':
      patch.plan = p;
      break;

    case 'repo_started':
      withRepo(evt.repo_id, p.repo, (r) => {
        r.status = 'migrating';
        pushActivity(r, {
          kind: 'step',
          tone: 'info',
          text: p.rule ? `Started migrating (rule ${p.rule})` : 'Started migrating',
        });
      });
      break;

    case 'step':
      withRepo(evt.repo_id, null, (r) => {
        r.steps += 1;
        pushActivity(r, { kind: 'step', tone: 'info', agent: evt.agent, text: p.text });
      });
      break;

    case 'tool_call':
      withRepo(evt.repo_id, null, (r) => {
        r.tool_calls += 1;
        const summary = p.summary || p.tool;
        pushActivity(r, {
          kind: 'tool',
          tone: 'info',
          agent: evt.agent,
          text: `${p.tool} -> ${summary}`,
        });
      });
      break;

    case 'edit_applied':
      withRepo(evt.repo_id, null, (r) => {
        if (p.path && !r.files_changed.includes(p.path)) {
          r.files_changed = [...r.files_changed, p.path];
        }
        pushActivity(r, {
          kind: 'edit',
          tone: 'info',
          text: p.bytes != null ? `${p.path} (${p.bytes} bytes)` : p.path,
        });
      });
      break;

    case 'tests_run':
      withRepo(evt.repo_id, null, (r) => {
        r.tests_passing = Boolean(p.passed);
        pushActivity(r, {
          kind: 'tests',
          tone: p.passed ? 'success' : 'error',
          text: p.passed ? 'Tests PASSED' : 'Tests FAILED',
        });
      });
      break;

    case 'review':
      withRepo(evt.repo_id, null, (r) => {
        r.status = 'reviewing';
        r.review_verdict = {
          approve: Boolean(p.approve),
          confidence: p.confidence,
          reasons: p.reasons || [],
          tampered_with_tests: Boolean(p.tampered_with_tests),
        };
        const parts = [`Review ${p.approve ? 'APPROVED' : 'REJECTED'}`];
        if (p.confidence != null) parts.push(`confidence ${Math.round(Number(p.confidence) * 100)}%`);
        if (p.tampered_with_tests) parts.push('tests tampered');
        pushActivity(r, {
          kind: 'review',
          tone: p.approve ? 'success' : 'error',
          text: parts.join(' - '),
        });
      });
      break;

    case 'awaiting_approval':
      withRepo(evt.repo_id, p.repo, (r) => {
        r.status = 'awaiting_approval';
        r.diff = p.diff ?? r.diff;
        r.summary = p.summary ?? r.summary;
        if (Array.isArray(p.files_changed) && p.files_changed.length) {
          r.files_changed = p.files_changed;
        }
        pushActivity(r, { kind: 'await', tone: 'warning', text: 'Awaiting human approval' });
      });
      patch.pendingApprovals = {
        ...(patch.pendingApprovals || state.pendingApprovals),
        [evt.repo_id]: { repo_id: evt.repo_id, ...p },
      };
      break;

    case 'pr_opened':
      withRepo(evt.repo_id, null, (r) => {
        r.status = 'pr_open';
        r.pr = { number: p.number, title: p.title, url: p.url, branch: p.branch, kind: p.kind };
        pushActivity(r, {
          kind: 'pr',
          tone: 'success',
          text: `PR #${p.number} opened${p.title ? `: ${p.title}` : ''}`,
        });
      });
      break;

    case 'repo_done': {
      const id = p.repo_id || evt.repo_id;
      withRepo(id, p.name, (r) => {
        if (p.status) r.status = p.status;
        if (p.tests_passing != null) r.tests_passing = Boolean(p.tests_passing);
        if (Array.isArray(p.files_changed)) r.files_changed = p.files_changed;
        if (typeof p.steps === 'number') r.steps = p.steps;
        if (typeof p.tool_calls === 'number') r.tool_calls = p.tool_calls;
        if (p.cost_usd != null) r.cost_usd = Number(p.cost_usd);
        if (p.reason && ['escalated', 'failed', 'rejected'].includes(r.status)) {
          r.escalation_reason = p.reason;
        }
        pushActivity(r, {
          kind: 'done',
          tone: doneTone(p.status),
          text: `Finished: ${(p.status || '').replace(/_/g, ' ')}${p.reason ? ` (${p.reason})` : ''}`,
        });
      });
      // Clear any pending approval now that the repo is finalized.
      if (id) {
        const pa = { ...(patch.pendingApprovals || state.pendingApprovals) };
        delete pa[id];
        patch.pendingApprovals = pa;
      }
      break;
    }

    case 'guardrail_block':
      withRepo(evt.repo_id, null, (r) => {
        r.guardrail_blocks = [
          ...r.guardrail_blocks,
          { tool: p.tool, reason: p.reason, target: p.target },
        ];
        pushActivity(r, {
          kind: 'guardrail',
          tone: 'warning',
          text: `Guardrail blocked ${p.tool}${p.target ? ` on ${p.target}` : ''}: ${p.reason}`,
        });
      });
      break;

    case 'usage':
      patch.totalTokens = state.totalTokens + Number(p.tokens || 0);
      patch.totalCostUsd = state.totalCostUsd + Number(p.cost_usd || 0);
      if (evt.repo_id) {
        withRepo(evt.repo_id, null, (r) => {
          r.tokens += Number(p.tokens || 0);
          r.cost_usd += Number(p.cost_usd || 0);
        });
      }
      break;

    case 'job_summary':
      patch.summary = p;
      patch.jobStatus = 'done';
      if (p.total_tokens != null) patch.totalTokens = Number(p.total_tokens);
      if (p.total_cost_usd != null) patch.totalCostUsd = Number(p.total_cost_usd);
      if (p.mode) patch.mode = p.mode;
      break;

    case 'error':
      patch.error = p.message || 'The migration reported an error';
      // A job-level error (no repo_id) means the run crashed: mark it terminal so
      // the stream closes instead of reconnecting/replaying forever.
      if (!evt.repo_id) patch.jobStatus = 'failed';
      break;

    default:
      break;
  }

  return patch;
}

export const useMigrationStore = create((set, get) => ({
  // Catalog
  jobs: [],
  jobsLoading: false,
  jobsError: null,

  // System
  health: null,
  mode: null,

  // Active run
  runId: null,
  run: null, // full object once fetched (history replay)
  triggering: false,
  streamConnected: false,
  error: null,

  // Derived projection of the event stream
  ...emptyDerived(),

  // Approval submission state, keyed by repo_id
  deciding: {},

  // -------- Health --------
  loadHealth: async () => {
    try {
      const data = await systemAPI.getHealth();
      set({ health: data || null, mode: data?.mode || null });
    } catch {
      set({ health: { status: 'down', mode: null }, mode: null });
    }
  },

  // -------- Jobs --------
  loadJobs: async () => {
    set({ jobsLoading: true, jobsError: null });
    try {
      const data = await jobAPI.getJobs();
      set({ jobs: data.jobs || [], jobsLoading: false });
    } catch (err) {
      set({
        jobsError: err?.message || 'Failed to load jobs',
        jobsLoading: false,
      });
    }
  },

  // -------- Trigger + stream --------
  triggerRun: async ({ jobId, autoApprove }) => {
    set({ triggering: true, error: null });
    try {
      // Tear down any previous stream and reset the projection.
      get().reset();
      const { run_id: runId } = await runAPI.triggerRun({ jobId, autoApprove });
      set({ runId, jobStatus: 'planning', triggering: false });
      get().connectStream(runId);
      return runId;
    } catch (err) {
      set({
        error: err?.response?.data?.detail || err?.message || 'Failed to start the migration run',
        triggering: false,
      });
      return null;
    }
  },

  connectStream: (runId) => {
    if (unsubscribeStream) {
      unsubscribeStream();
      unsubscribeStream = null;
    }
    set({ runId, streamConnected: false });
    unsubscribeStream = subscribeToRun(runId, (evt) => get().pushEvent(evt), {
      // The backend replays the FULL event history on every (re)connection, so the
      // derived projection must be rebuilt from scratch each time the stream opens -
      // otherwise a transient reconnect doubles every counter and activity entry.
      onOpen: () => set({ streamConnected: true, ...emptyDerived() }),
      onError: () => set({ streamConnected: false }),
    });
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
    set((state) => reduceEvent(state, evt));

    // Auto close the stream once the run reaches a terminal status.
    if (TERMINAL_JOB_STATUSES.includes(get().jobStatus)) {
      get().disconnectStream();
    }
  },

  // -------- Approval gate --------
  // Multiple repos can await approval at once, so decisions are keyed by repo_id.
  decide: async (repoId, decision, note) => {
    const { runId } = get();
    if (!runId || !repoId) return;
    set((s) => ({ deciding: { ...s.deciding, [repoId]: true } }));
    try {
      await runAPI.decide(runId, { repoId, decision, note });
      // Optimistically drop the pending approval; the stream confirms via
      // pr_opened / repo_done. Keep the repo lane so the operator sees progress.
      set((s) => {
        const pending = { ...s.pendingApprovals };
        delete pending[repoId];
        const deciding = { ...s.deciding };
        delete deciding[repoId];
        return { pendingApprovals: pending, deciding };
      });
    } catch (err) {
      set((s) => {
        const deciding = { ...s.deciding };
        delete deciding[repoId];
        return {
          deciding,
          error:
            err?.response?.data?.detail || err?.message || 'Failed to submit the approval decision',
        };
      });
    }
  },

  // -------- Loading a stored run (history replay) --------
  loadRun: async (runId) => {
    set({ error: null });
    try {
      const data = await runAPI.getRun(runId);
      set({ runId, run: data, ...projectRun(data), pendingApprovals: {}, streamConnected: false });
      return data;
    } catch (err) {
      set({ error: err?.message || 'Failed to load the run' });
      return null;
    }
  },

  // -------- Reset --------
  reset: () => {
    if (unsubscribeStream) {
      unsubscribeStream();
      unsubscribeStream = null;
    }
    set({
      runId: null,
      run: null,
      triggering: false,
      streamConnected: false,
      error: null,
      deciding: {},
      ...emptyDerived(),
    });
  },
}));

/**
 * Build the full derived projection of a stored run object. Pure and
 * store-independent, so both loadRun() and the read-only history view use it.
 * Replays the recorded events through the live reducer, then backfills
 * authoritative final fields from the stored RepoState objects.
 */
export function projectRun(data) {
  let derived = emptyDerived();
  (data.events || []).forEach((evt) => {
    derived = { ...derived, ...reduceEvent(derived, evt) };
  });

  const repos = { ...derived.repos };
  const reposOrder = [...derived.reposOrder];
  (data.repos || []).forEach((rs) => {
    const cur = repos[rs.repo_id];
    if (!cur) {
      repos[rs.repo_id] = fromRepoState(rs);
      reposOrder.push(rs.repo_id);
    } else {
      repos[rs.repo_id] = {
        ...cur,
        status: cur.status || rs.status,
        diff: cur.diff ?? rs.diff ?? null,
        summary: cur.summary ?? rs.summary ?? null,
        pr: cur.pr || rs.pr || null,
        review_verdict: cur.review_verdict || rs.review_verdict || null,
        escalation_reason: cur.escalation_reason || rs.escalation_reason || null,
        files_changed: cur.files_changed.length ? cur.files_changed : rs.files_changed || [],
        tests_passing: cur.tests_passing ?? rs.tests_passing ?? null,
      };
    }
  });

  return {
    title: derived.title || data.title || null,
    mode: derived.mode || data.mode || null,
    plan: derived.plan,
    jobStatus: derived.jobStatus || data.status || 'done',
    repos,
    reposOrder,
    events: derived.events,
    summary: derived.summary || fallbackSummary(repos, data),
    totalTokens: data.total_tokens ?? derived.totalTokens,
    totalCostUsd: data.total_cost_usd ?? derived.totalCostUsd,
  };
}

// Build a repo lane directly from a stored RepoState (no event history).
function fromRepoState(rs) {
  return {
    repo_id: rs.repo_id,
    name: rs.name || rs.repo_id,
    status: rs.status || 'queued',
    steps: typeof rs.steps === 'number' ? rs.steps : 0,
    tool_calls: typeof rs.tool_calls === 'number' ? rs.tool_calls : 0,
    files_changed: rs.files_changed || [],
    tokens: rs.tokens || 0,
    cost_usd: rs.cost_usd || 0,
    tests_passing: rs.tests_passing ?? null,
    review_verdict: rs.review_verdict || null,
    diff: rs.diff || null,
    summary: rs.summary || null,
    pr: rs.pr || null,
    guardrail_blocks: rs.guardrail_blocks || [],
    escalation_reason: rs.escalation_reason || null,
    activity: [],
  };
}

// Compose a job summary when the stored run has no job_summary event.
function fallbackSummary(repos, data) {
  const list = Object.values(repos);
  const by_status = {};
  let merged = 0;
  list.forEach((r) => {
    by_status[r.status] = (by_status[r.status] || 0) + 1;
    if (r.status === 'pr_open') merged += 1;
  });
  return {
    repos_total: list.length,
    repos_merged: merged,
    by_status,
    total_tokens: data.total_tokens || 0,
    total_cost_usd: data.total_cost_usd || 0,
    mode: data.mode || null,
  };
}
