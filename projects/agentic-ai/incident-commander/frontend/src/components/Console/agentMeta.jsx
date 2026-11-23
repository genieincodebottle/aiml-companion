// Shared visual metadata for agents and event types.
// Keeps color-coding consistent across the timeline, the flow view and history.
//
// Palette rule: blue/teal for neutral chrome, amber for "a human should look",
// green for good, red for bad, slate for bookkeeping. No violet or magenta - a
// color in this console is supposed to mean something, and a decorative hue that
// means nothing trains people to ignore all of them.

import {
  TravelExplore,
  Biotech,
  Build,
  FactCheck,
  Description,
  Info,
  PlayArrow,
  Stop,
  Lightbulb,
  Science,
  Gavel,
  CheckCircle,
  ReportProblem,
  Loop,
  Token,
  ErrorOutlineOutlined as ErrorOutline,
  Bolt,
  Balance,
  AltRoute,
  Psychology,
  TravelExplore as SearchIcon,
  Shield,
  History,
  Block,
} from '@mui/icons-material';

// Agent identity -> color + icon + a plain-English job description.
// `does` is rendered in the UI, so a reader never has to guess what "Critic" means.
export const AGENT_META = {
  Supervisor: {
    color: '#a3e635',
    icon: <AltRoute fontSize="small" />,
    does: 'Decides which specialist runs next.',
  },
  Triage: {
    color: '#38bdf8',
    icon: <TravelExplore fontSize="small" />,
    does: 'Sets severity and blast radius. Runs once.',
  },
  Diagnostician: {
    color: '#22d3ee',
    icon: <Biotech fontSize="small" />,
    does: 'Ranks root-cause hypotheses, or asks for more telemetry.',
  },
  Remediation: {
    color: '#fbbf24',
    icon: <Build fontSize="small" />,
    does: 'Proposes exactly one fix, with a prediction that can be proven wrong.',
  },
  Critic: {
    color: '#fb923c',
    icon: <Balance fontSize="small" />,
    does: 'Argues against that fix before a human is asked to approve it.',
  },
  Verifier: {
    color: '#34d399',
    icon: <FactCheck fontSize="small" />,
    does: 'Reads fresh telemetry and judges whether the prediction came true.',
  },
  Postmortem: {
    color: '#60a5fa',
    icon: <Description fontSize="small" />,
    does: 'Writes the report and stores a reusable lesson in memory.',
  },
  Orchestrator: {
    color: '#94a3b8',
    icon: <Loop fontSize="small" />,
    does: 'The loop itself: ceilings, memory recall, halt conditions.',
  },
};

const DEFAULT_AGENT = { color: '#94a3b8', icon: <Bolt fontSize="small" />, does: '' };

export function agentMeta(agent) {
  return AGENT_META[agent] || DEFAULT_AGENT;
}

// Event type -> label + icon + fallback accent + a one-line explanation.
// `hint` answers "why am I looking at this?" for someone reading the console for
// the first time. It is shown in the flow view's detail panel.
export const EVENT_META = {
  status: {
    label: 'Status',
    icon: <Info fontSize="small" />,
    color: '#94a3b8',
    hint: 'A state transition in the incident.',
  },
  step_start: {
    label: 'Thinking',
    icon: <PlayArrow fontSize="small" />,
    color: '#4f9eff',
    hint: 'An agent started work. The model call is in flight.',
  },
  step_end: {
    label: 'Step done',
    icon: <Stop fontSize="small" />,
    color: '#64748b',
    hint: 'An agent returned a structured result.',
  },
  hypothesis: {
    label: 'Hypothesis',
    icon: <Lightbulb fontSize="small" />,
    color: '#22d3ee',
    hint: 'One candidate root cause, with the confidence the model assigned it.',
  },
  diagnostic: {
    label: 'Diagnosis',
    icon: <Science fontSize="small" />,
    color: '#22d3ee',
    hint: 'The leading cause and the evidence behind it. Below the confidence floor the loop refuses to act.',
  },
  proposed_action: {
    label: 'Proposed fix',
    icon: <Build fontSize="small" />,
    color: '#fbbf24',
    hint: 'One action from a closed menu, plus a falsifiable prediction. The risk shown was assigned by code, not by the model.',
  },
  awaiting_approval: {
    label: 'Waiting for you',
    icon: <Gavel fontSize="small" />,
    color: '#fbbf24',
    hint: 'The loop has stopped. Nothing runs until a human decides.',
  },
  action_executed: {
    label: 'Action executed',
    icon: <Bolt fontSize="small" />,
    color: '#fbbf24',
    hint: 'The only mutating call in the system just ran against the sandbox.',
  },
  verification: {
    label: 'Verification',
    icon: <FactCheck fontSize="small" />,
    color: '#34d399',
    hint: 'Fresh telemetry, checked against the prediction made before the action.',
  },
  loop: {
    label: 'New iteration',
    icon: <Loop fontSize="small" />,
    color: '#38bdf8',
    hint: 'The previous pass did not resolve it, so the loop goes round again - bounded by the iteration ceiling.',
  },
  resolved: {
    label: 'Resolved',
    icon: <CheckCircle fontSize="small" />,
    color: '#34d399',
    hint: 'The Verifier confirmed recovery.',
  },
  escalated: {
    label: 'Escalated',
    icon: <ReportProblem fontSize="small" />,
    color: '#f87171',
    hint: 'The agent stopped and handed over. For some scenarios this is the correct answer.',
  },
  postmortem: {
    label: 'Postmortem',
    icon: <Description fontSize="small" />,
    color: '#60a5fa',
    hint: 'Written for every incident, resolved or not, and stored as a retrievable lesson.',
  },
  token_usage: {
    label: 'Cost',
    icon: <Token fontSize="small" />,
    color: '#64748b',
    hint: 'Cumulative spend for this incident. Checked against the ceiling before every model call.',
  },
  error: {
    label: 'Error',
    icon: <ErrorOutline fontSize="small" />,
    color: '#f87171',
    hint: 'Something failed. The loop records it rather than pretending it did not happen.',
  },
  route: {
    label: 'Routed',
    icon: <AltRoute fontSize="small" />,
    color: '#a3e635',
    hint: 'Who acts next. Whatever the model suggests, code clamps the choice to a legal one.',
  },
  critique: {
    label: 'Critique',
    icon: <Balance fontSize="small" />,
    color: '#fb923c',
    hint: 'A second agent, told to find the flaw. A veto sends the loop back to re-diagnose.',
  },
  evidence_request: {
    label: 'Needs more evidence',
    icon: <SearchIcon fontSize="small" />,
    color: '#22d3ee',
    hint: 'The agent said it cannot commit yet and named the tools it wants. This is what "not sure" is wired to.',
  },
  evidence_gathered: {
    label: 'Evidence gathered',
    icon: <Psychology fontSize="small" />,
    color: '#22d3ee',
    hint: 'Those read-only tools ran. Their output is sanitized like any other telemetry.',
  },
  memory_recall: {
    label: 'Recalled from memory',
    icon: <History fontSize="small" />,
    color: '#60a5fa',
    hint: 'Similar past incidents, retrieved before the first diagnosis.',
  },
  guardrail: {
    label: 'Guardrail fired',
    icon: <Shield fontSize="small" />,
    color: '#fb923c',
    hint: 'A rule enforced in code, not in a prompt.',
  },
  cancelled: {
    label: 'Cancelled',
    icon: <Block fontSize="small" />,
    color: '#f87171',
    hint: 'An operator stopped the loop. A real halt condition, not a UI state.',
  },
};

export function eventMeta(type) {
  return (
    EVENT_META[type] || {
      label: type || 'Event',
      icon: <Info fontSize="small" />,
      color: '#94a3b8',
      hint: '',
    }
  );
}

// The canonical stages of one pass through the loop, in order. Drives the flow
// diagram and the "what is happening now" strip, so both stay in step with the
// timeline instead of drifting into their own vocabulary.
export const LOOP_STAGES = [
  {
    key: 'observe',
    label: 'Observe',
    agent: 'Orchestrator',
    blurb: 'Read-only telemetry is collected and sanitized before any agent sees it.',
    events: ['status', 'memory_recall'],
  },
  {
    key: 'route',
    label: 'Route',
    agent: 'Supervisor',
    blurb: 'Pick the next specialist. Code clamps the choice to a legal one.',
    events: ['route'],
  },
  {
    key: 'triage',
    label: 'Triage',
    agent: 'Triage',
    blurb: 'Severity and blast radius. Runs once per incident.',
    events: [],
  },
  {
    key: 'diagnose',
    label: 'Diagnose',
    agent: 'Diagnostician',
    blurb: 'Ranked hypotheses. If confidence is too low, ask for tools instead of guessing.',
    events: ['hypothesis', 'diagnostic', 'evidence_request', 'evidence_gathered'],
  },
  {
    key: 'propose',
    label: 'Propose',
    agent: 'Remediation',
    blurb: 'One action from a closed menu, with a prediction that can be proven wrong.',
    events: ['proposed_action'],
  },
  {
    key: 'critique',
    label: 'Critique',
    agent: 'Critic',
    blurb: 'A second agent argues against the fix before a human is bothered.',
    events: ['critique'],
  },
  {
    key: 'approve',
    label: 'Human gate',
    agent: null,
    blurb: 'High-risk actions always stop here. The loop genuinely waits.',
    events: ['awaiting_approval', 'guardrail'],
  },
  {
    key: 'act',
    label: 'Act',
    agent: null,
    blurb: 'The single mutating call in the system.',
    events: ['action_executed'],
  },
  {
    key: 'verify',
    label: 'Verify',
    agent: 'Verifier',
    blurb: 'Fresh telemetry. Did the predicted signal actually move?',
    events: ['verification'],
  },
  {
    key: 'finish',
    label: 'Resolve / escalate',
    agent: 'Postmortem',
    blurb: 'Every incident ends with a typed halt reason and a stored postmortem.',
    events: ['resolved', 'escalated', 'cancelled', 'postmortem'],
  },
];

// Which stage an event belongs to. Used to light up the flow diagram live.
const STAGE_BY_EVENT = (() => {
  const map = {};
  LOOP_STAGES.forEach((stage) =>
    stage.events.forEach((e) => {
      if (!map[e]) map[e] = stage.key;
    })
  );
  return map;
})();

const STAGE_BY_AGENT = (() => {
  const map = {};
  LOOP_STAGES.forEach((stage) => {
    if (stage.agent && !map[stage.agent]) map[stage.agent] = stage.key;
  });
  return map;
})();

export function stageForEvent(event) {
  if (!event) return null;
  // step_start/step_end carry the agent that is working, which is a better signal
  // for "where are we now" than the generic event type.
  if (event.type === 'step_start' || event.type === 'step_end') {
    return STAGE_BY_AGENT[event.agent] || null;
  }
  return STAGE_BY_EVENT[event.type] || null;
}

// How prominent a step should look.
//
// Borrowed from a pattern worth stealing: instead of every row being visually
// equal, the state drives fill, border weight and elevation, so "the loop is
// waiting on you" is legible from across the room rather than requiring you to
// read the label. Four states only - more would be decoration.
//
//   'waiting'  a human must act. The loudest thing on the page.
//   'good'     resolved, recovered, endorsed.
//   'bad'      escalated, cancelled, errored, vetoed.
//   'active'   a model call is in flight.
//   null       ordinary reasoning; falls back to the agent's accent.
export function eventTone(event) {
  const { type, payload = {} } = event || {};
  switch (type) {
    case 'awaiting_approval':
      return 'waiting';
    case 'resolved':
      return 'good';
    case 'escalated':
    case 'cancelled':
    case 'error':
      return 'bad';
    case 'step_start':
      return 'active';
    case 'verification':
      return payload.recovered ? 'good' : 'waiting';
    case 'critique':
      // `endorse` is a boolean, so check for an explicit false rather than
      // falsiness - an absent field must not read as a veto.
      return payload.endorse === false ? 'bad' : null;
    case 'guardrail':
      return 'waiting';
    case 'action_executed':
      return payload.harmful ? 'bad' : null;
    default:
      return null;
  }
}

// Pure helpers live in a dependency-free module; re-exported here so existing
// imports (`from './agentMeta'`) keep working.
export { riskColor, statusColor, formatTime } from './format';
