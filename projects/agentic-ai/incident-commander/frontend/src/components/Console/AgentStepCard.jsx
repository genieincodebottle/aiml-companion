import { useState } from 'react';
import {
  Box,
  Chip,
  Collapse,
  IconButton,
  LinearProgress,
  Paper,
  Stack,
  Tooltip,
  Typography,
} from '@mui/material';
import { ExpandMore, ExpandLess, HelpOutlineOutlined as HelpOutline } from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import { agentMeta, eventMeta, eventTone, riskColor, formatTime } from './agentMeta';
import { toneFor, tintFor } from '../../theme';

// Renders one SSE event as a step card.
//
// The design goal is that nothing the agent decided is hidden. Every event type
// the backend emits has a body here, because an event rendered as a bare title is
// indistinguishable to a reader from an agent that did nothing - which is exactly
// the "black box" complaint this component exists to answer. If you add an event
// type to the backend, add a case below; the default branch dumps the raw payload
// rather than silently swallowing it.
export default function AgentStepCard({ event, defaultExpanded = false }) {
  const theme = useTheme();
  const { type, agent, ts } = event || {};
  const mode = theme.palette.mode;
  const evt = eventMeta(type);
  const am = agent ? agentMeta(agent) : null;
  // Accents are authored for dark; darken them so they stay legible on white.
  const accent = toneFor(am ? am.color : evt.color, mode);
  const accentTint = tintFor(am ? am.color : evt.color, mode);
  const icon = am ? am.icon : evt.icon;
  const [open, setOpen] = useState(defaultExpanded);

  // A step that needs attention, or that ended the run, is drawn loudly: filled
  // status circle, tinted surface, 2px semantic border. Ordinary reasoning steps
  // stay quiet on the agent's accent. Making every row equally emphatic is the
  // same as making none of them emphatic.
  const tone = eventTone(event);
  const toned = TONES(theme)[tone];
  const circleBg = toned ? toned.solid : accentTint;
  const circleFg = toned ? toned.on : accent;

  const extra = extraDetail(event);

  return (
    <Paper
      elevation={tone === 'waiting' ? 3 : 0}
      variant={toned ? 'elevation' : 'outlined'}
      sx={{
        p: 1.5,
        borderRadius: 2,
        border: toned ? 2 : 1,
        borderColor: toned ? toned.border : 'divider',
        borderLeft: toned ? undefined : `3px solid ${accent}`,
        bgcolor: toned ? toned.bg : 'background.paper',
        transition: 'box-shadow 200ms ease-out, border-color 200ms ease-out',
        '&:hover': { boxShadow: toned ? 4 : 1 },
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.25, mb: 0.75 }}>
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: toned ? 34 : 26,
            height: toned ? 34 : 26,
            borderRadius: '50%',
            bgcolor: circleBg,
            color: circleFg,
            boxShadow: toned ? 2 : 0,
            flexShrink: 0,
          }}
        >
          {icon}
        </Box>
        <Typography
          variant="subtitle2"
          sx={{
            color: toned ? toned.text : accent,
            fontWeight: tone === 'waiting' ? 800 : 600,
            flexGrow: 1,
          }}
        >
          {agent ? `${agent} - ${evt.label}` : evt.label}
        </Typography>
        {evt.hint && (
          <Tooltip title={evt.hint} placement="left">
            <HelpOutline sx={{ fontSize: 15, color: 'text.disabled' }} />
          </Tooltip>
        )}
        {ts && (
          <Typography variant="caption" color="text.secondary">
            {formatTime(ts)}
          </Typography>
        )}
        {extra && (
          <IconButton size="small" onClick={() => setOpen((v) => !v)} sx={{ p: 0.25 }}>
            {open ? <ExpandLess fontSize="small" /> : <ExpandMore fontSize="small" />}
          </IconButton>
        )}
      </Box>

      <EventBody event={event} accent={accent} />

      {extra && (
        <Collapse in={open}>
          <Box sx={{ mt: 1 }}>{extra}</Box>
        </Collapse>
      )}
    </Paper>
  );
}

// ---------------------------------------------------------------------------
// small shared pieces
// ---------------------------------------------------------------------------

// Surface treatment per tone. Reads from the palette rather than hardcoding hex,
// so light and dark both get a legible fill without a second table.
function TONES(theme) {
  const a = theme.palette.mode === 'light' ? 0.07 : 0.12;
  const rgba = (hex, alpha) => {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
  };
  const make = (p) => ({
    solid: p.main,
    on: theme.palette.getContrastText(p.main),
    border: p.main,
    bg: rgba(p.main, a),
    text: theme.palette.mode === 'light' ? p.dark : p.light,
  });
  return {
    waiting: make(theme.palette.warning),
    good: make(theme.palette.success),
    bad: make(theme.palette.error),
    active: make(theme.palette.info),
  };
}

function Line({ label, value, mono }) {
  if (value == null || value === '') return null;
  return (
    <Typography
      variant="body2"
      color="text.secondary"
      sx={{ wordBreak: 'break-word', fontFamily: mono ? 'ui-monospace, monospace' : undefined }}
    >
      {label && (
        <Typography
          component="span"
          variant="body2"
          sx={{ color: 'text.primary', fontWeight: 600 }}
        >
          {label}:{' '}
        </Typography>
      )}
      {String(value)}
    </Typography>
  );
}

function Bullets({ label, items }) {
  const list = (items || []).filter(Boolean);
  if (!list.length) return null;
  return (
    <Box>
      {label && (
        <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.25 }}>
          {label}
        </Typography>
      )}
      <Box component="ul" sx={{ m: 0, pl: 2.5 }}>
        {list.map((it, i) => (
          <Typography key={i} component="li" variant="body2" color="text.secondary">
            {typeof it === 'string' ? it : JSON.stringify(it)}
          </Typography>
        ))}
      </Box>
    </Box>
  );
}

// Confidence is the single most useful number in this UI and was previously
// rendered as bare text. "0.55 against a 0.6 floor" is a fact about whether the
// agent is *allowed* to act, so it gets a bar rather than a sentence.
function Confidence({ value, floor }) {
  if (value == null) return null;
  const pct = Math.round(Number(value) * 100);
  const below = floor != null && Number(value) < Number(floor);
  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 1 }}>
        <Typography variant="body2" sx={{ fontWeight: 600 }}>
          Confidence {pct}%
        </Typography>
        {floor != null && (
          <Typography variant="caption" color={below ? 'warning.main' : 'text.secondary'}>
            {below
              ? `below the ${Math.round(floor * 100)}% action floor`
              : `floor ${Math.round(floor * 100)}%`}
          </Typography>
        )}
      </Box>
      <LinearProgress
        variant="determinate"
        value={Math.min(100, pct)}
        color={below ? 'warning' : pct >= 75 ? 'success' : 'info'}
        sx={{ height: 5, borderRadius: 3, mt: 0.4 }}
      />
    </Box>
  );
}

function Raw({ data }) {
  return (
    <Box
      component="pre"
      sx={{
        m: 0,
        p: 1,
        maxHeight: 260,
        overflow: 'auto',
        fontSize: 11,
        lineHeight: 1.5,
        borderRadius: 1,
        bgcolor: 'action.hover',
        border: '1px solid',
        borderColor: 'divider',
        color: 'text.secondary',
        whiteSpace: 'pre-wrap',
      }}
    >
      {JSON.stringify(data, null, 2)}
    </Box>
  );
}

// Bulky detail worth keeping but not worth showing by default.
function extraDetail(event) {
  const { type, payload = {} } = event || {};
  if (type === 'evidence_gathered' && payload.results) return <Raw data={payload.results} />;
  if (type === 'memory_recall' && payload.episodes?.length) return <Raw data={payload.episodes} />;
  if (type === 'postmortem') return <Raw data={payload} />;
  return null;
}

// ---------------------------------------------------------------------------
// per-event bodies
// ---------------------------------------------------------------------------

function EventBody({ event, accent }) {
  const { type, payload = {} } = event || {};

  switch (type) {
    case 'status':
      return (
        <Stack spacing={0.5}>
          <Line value={payload.message || payload.status} />
          {payload.max_iterations != null && (
            <Line
              label="ceilings"
              value={`${payload.max_iterations} iterations, $${payload.max_cost_usd}`}
            />
          )}
        </Stack>
      );

    case 'step_start':
      return (
        <Stack spacing={0.5}>
          <Typography variant="body2" color="text.secondary">
            {payload.re_diagnosis
              ? 'Re-running the diagnosis with the new evidence.'
              : 'Model call in flight...'}
          </Typography>
          <LinearProgress sx={{ height: 3, borderRadius: 3 }} />
        </Stack>
      );

    case 'step_end':
      return (
        <Stack spacing={0.5}>
          <Line value={payload.name || payload.step} />
          <Line label="blast radius" value={payload.blast_radius || payload.message} />
          {payload.customer_facing != null && (
            <Chip
              size="small"
              variant="outlined"
              color={payload.customer_facing ? 'error' : 'default'}
              label={payload.customer_facing ? 'customer facing' : 'internal only'}
              sx={{ alignSelf: 'flex-start' }}
            />
          )}
        </Stack>
      );

    case 'route':
      return (
        <Stack spacing={0.5}>
          <Line label="next" value={payload.route} />
          <Line label="why" value={payload.reason} />
          <Chip
            size="small"
            variant="outlined"
            label={payload.llm ? 'chosen by the LLM Supervisor' : 'deterministic policy'}
            sx={{ alignSelf: 'flex-start' }}
          />
        </Stack>
      );

    case 'memory_recall':
      return (
        <Stack spacing={0.5}>
          <Line value={payload.message} />
          <Bullets items={(payload.episodes || []).map((e) => e.prompt_line || e.summary)} />
          <Typography variant="caption" color="text.secondary">
            Expand for the full retrieved episodes and their retrieval scores.
          </Typography>
        </Stack>
      );

    case 'hypothesis':
      return (
        <Stack spacing={0.75}>
          <Line value={payload.cause || payload.text} />
          <Confidence value={payload.confidence} />
          <Line label="would be confirmed by" value={payload.tool_to_check} />
          <Line label="evidence needed" value={payload.evidence_needed} />
        </Stack>
      );

    case 'diagnostic':
      return (
        <Stack spacing={0.75}>
          <Typography variant="body2" sx={{ color: 'text.primary' }}>
            {payload.leading_cause || payload.finding}
          </Typography>
          <Confidence value={payload.confidence} floor={payload.confidence_floor} />
          <Bullets label="Evidence used" items={payload.evidence} />
          {payload.used_recalled_episode && (
            <Chip
              size="small"
              variant="outlined"
              color="info"
              label="used a recalled past incident"
              sx={{ alignSelf: 'flex-start' }}
            />
          )}
          {payload.needs_more_data && (
            <Chip
              size="small"
              color="warning"
              variant="outlined"
              label={`wants: ${(payload.requested_tools || []).join(', ') || 'more data'}`}
              sx={{ alignSelf: 'flex-start' }}
            />
          )}
        </Stack>
      );

    case 'evidence_request':
      return (
        <Stack spacing={0.75}>
          <Line value={payload.reason} />
          <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
            {(payload.tools || []).map((t) => (
              <Chip key={t} size="small" variant="outlined" label={t} />
            ))}
          </Box>
          <Typography variant="caption" color="text.secondary">
            Evidence round {payload.round} of {payload.max_rounds} allowed
            {payload.below_confidence_floor ? ' - confidence was below the action floor' : ''}
          </Typography>
        </Stack>
      );

    case 'evidence_gathered':
      return (
        <Stack spacing={0.5}>
          <Line label="ran" value={(payload.tools || []).join(', ')} />
          <Typography variant="caption" color="text.secondary">
            Expand to read exactly what came back - this is what the next diagnosis sees.
          </Typography>
        </Stack>
      );

    case 'proposed_action':
      return (
        <Stack spacing={0.75}>
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
            <Chip
              size="small"
              label={payload.action_type}
              sx={{ bgcolor: `${accent}22`, color: accent }}
            />
            {payload.risk && (
              <Chip size="small" color={riskColor(payload.risk)} label={`${payload.risk} risk`} />
            )}
            {payload.reversible != null && (
              <Chip
                size="small"
                variant="outlined"
                label={payload.reversible ? 'reversible' : 'not reversible'}
              />
            )}
          </Box>
          <Line label="target" value={payload.target} />
          <Line label="reasoning" value={payload.rationale} />
          <Line label="predicts" value={payload.expected_signal} />
          <Line label="fallback if ineffective" value={payload.fallback} />
          {payload.self_assessed_risk && payload.self_assessed_risk !== payload.risk && (
            <Typography variant="caption" sx={{ color: 'warning.main' }}>
              The model graded this {payload.self_assessed_risk} risk. Code overrode it to{' '}
              {payload.risk} - an agent that grades its own risk could route around the approval
              gate.
            </Typography>
          )}
        </Stack>
      );

    case 'critique':
      return (
        <Stack spacing={0.75}>
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
            <Chip
              size="small"
              color={payload.endorse ? 'success' : 'error'}
              label={payload.endorse ? 'endorsed' : 'VETOED'}
            />
            {payload.safer_alternative && (
              <Chip
                size="small"
                variant="outlined"
                label={`prefers: ${payload.safer_alternative}`}
              />
            )}
          </Box>
          <Line label="strongest objection" value={payload.strongest_objection} />
          <Bullets label="Concerns" items={payload.concerns} />
          <Confidence value={payload.confidence} />
          {payload.endorse === false && (
            <Typography variant="caption" sx={{ color: 'warning.main' }}>
              A veto sends the loop back to re-diagnose rather than to a human.
            </Typography>
          )}
        </Stack>
      );

    case 'guardrail':
      return (
        <Stack spacing={0.5}>
          <Line label={payload.kind || 'rule'} value={payload.message || payload.detail} />
          {payload.patterns?.length > 0 && (
            <Line label="patterns matched" value={payload.patterns.join(', ')} mono />
          )}
          <Line label="action" value={payload.action_type} />
        </Stack>
      );

    case 'awaiting_approval':
      return (
        <Stack spacing={0.75}>
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
            <Chip
              size="small"
              label={payload.action_type}
              sx={{ bgcolor: `${accent}22`, color: accent }}
            />
            {payload.risk && (
              <Chip size="small" color={riskColor(payload.risk)} label={`${payload.risk} risk`} />
            )}
          </Box>
          <Line label="target" value={payload.target} />
          <Line label="reasoning" value={payload.rationale} />
          <Line label="predicts" value={payload.expected_signal} />
          {payload.critique && (
            <Line
              label="the Critic said"
              value={
                payload.critique.strongest_objection ||
                (payload.critique.endorse ? 'endorsed it' : '')
              }
            />
          )}
          <Typography variant="caption" sx={{ color: 'warning.main', fontWeight: 600 }}>
            The loop is stopped and waiting. Decide in the Approval panel
            {payload.timeout_s
              ? ` within ${Math.round(payload.timeout_s / 60)} min, or it escalates`
              : ''}
            .
          </Typography>
        </Stack>
      );

    case 'action_executed':
      return (
        <Stack spacing={0.5}>
          <Line label="action" value={payload.action_type} />
          <Line label="target" value={payload.target} />
          <Line label="result" value={payload.result || payload.status || payload.reason} />
          <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
            {payload.effective != null && (
              <Chip
                size="small"
                variant="outlined"
                color={payload.effective ? 'success' : 'default'}
                label={payload.effective ? 'effective' : 'no effect'}
              />
            )}
            {payload.harmful && <Chip size="small" color="error" label="HARMFUL" />}
            {payload.progress && (
              <Chip size="small" variant="outlined" label={`progress: ${payload.progress}`} />
            )}
          </Box>
        </Stack>
      );

    case 'verification':
      return (
        <Stack spacing={0.75}>
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
            <Chip
              size="small"
              color={payload.recovered ? 'success' : 'warning'}
              label={payload.recovered ? 'recovered' : 'not recovered'}
            />
            {payload.expected_signal_met != null && (
              <Chip
                size="small"
                variant="outlined"
                color={payload.expected_signal_met ? 'success' : 'warning'}
                label={payload.expected_signal_met ? 'prediction held' : 'prediction did NOT hold'}
              />
            )}
          </Box>
          <Line label="signal" value={payload.signal_assessment} />
          <Bullets label="Observed" items={payload.observations} />
          <Line label="next step" value={payload.next_step} />
          <Line label="partial progress" value={payload.partial_progress} />
          <Confidence value={payload.confidence} />
        </Stack>
      );

    case 'loop':
      return <Line value={`Starting iteration ${payload.iteration}.`} />;

    case 'resolved':
      return (
        <Stack spacing={0.5}>
          <Line value={payload.message || 'Incident resolved by the agent loop.'} />
          <Line label="halt reason" value={payload.reason} mono />
        </Stack>
      );

    case 'escalated':
    case 'cancelled':
      return (
        <Stack spacing={0.5}>
          <Line value={payload.message || 'Handed to a human responder.'} />
          <Line label="halt reason" value={payload.reason} mono />
        </Stack>
      );

    case 'postmortem':
      return (
        <Stack spacing={0.5}>
          <Line label="summary" value={payload.summary} />
          <Line label="root cause" value={payload.root_cause} />
          <Line label="lesson" value={payload.runbook_lesson || payload.lesson} />
        </Stack>
      );

    case 'token_usage':
      return (
        <Stack spacing={0.5}>
          <Line
            value={`${payload.tokens ?? 0} tokens, $${Number(payload.cost_usd || 0).toFixed(4)} of $${
              payload.budget_usd ?? '-'
            }`}
          />
          {payload.budget_used_pct != null && (
            <LinearProgress
              variant="determinate"
              value={Math.min(100, Number(payload.budget_used_pct))}
              color={Number(payload.budget_used_pct) > 80 ? 'warning' : 'info'}
              sx={{ height: 4, borderRadius: 3 }}
            />
          )}
        </Stack>
      );

    case 'error':
      return <Line value={payload.message || 'Unknown error'} />;

    default:
      // Never swallow an event. An unrendered payload still tells a reader more
      // than a bare title does.
      return Object.keys(payload).length ? <Raw data={payload} /> : null;
  }
}
