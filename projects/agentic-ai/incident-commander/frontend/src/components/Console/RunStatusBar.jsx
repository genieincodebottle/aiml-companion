import { useEffect, useState } from 'react';
import {
  Box,
  Button,
  Card,
  Chip,
  CircularProgress,
  LinearProgress,
  Stack,
  Tooltip,
  Typography,
} from '@mui/material';
import { CheckCircle, ReportProblem, Block, Gavel, Stop } from '@mui/icons-material';
import { useIncidentStore, TERMINAL_STATUSES } from '../../store/incidentStore';
import { LOOP_STAGES, stageForEvent } from './agentMeta';

// The always-visible answer to "is anything actually happening?".
//
// This exists because of a specific, repeatable misreading of the old console: a
// model call takes 10-30 seconds, during which nothing moved on screen, so people
// concluded the trigger had failed and pressed it again. A spinner alone does not
// fix that - it says "busy" without saying "busy doing what, and for how long".
// So this bar carries four things at once: the named activity, elapsed seconds,
// which stage of the loop we are in, and what it is costing.
export default function RunStatusBar() {
  const {
    incidentId,
    status,
    haltReason,
    activity,
    agentBusy,
    startedAt,
    events,
    iteration,
    maxIterations,
    totalCostUsd,
    budgetUsd,
    streamConnected,
    pendingApproval,
    cancelIncident,
    cancelling,
  } = useIncidentStore();

  // Elapsed time ticks locally; the backend does not stream a clock, and a frozen
  // "0s" would recreate the exact problem this component solves.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!startedAt || TERMINAL_STATUSES.includes(status)) return undefined;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [startedAt, status]);

  if (!incidentId) return <IdleBar />;

  const terminal = TERMINAL_STATUSES.includes(status);
  // `now` ticks every second while running and freezes once the run is terminal.
  const elapsed = startedAt ? Math.max(0, Math.round((now - startedAt) / 1000)) : 0;
  const waiting = status === 'awaiting_approval' || Boolean(pendingApproval);

  // Which stage the most recent meaningful event belongs to.
  const currentStage = (() => {
    for (let i = events.length - 1; i >= 0; i -= 1) {
      const s = stageForEvent(events[i]);
      if (s) return s;
    }
    return null;
  })();

  const tone = terminal
    ? status === 'resolved'
      ? { color: 'success.main', bg: 'rgba(52,211,153,0.10)', border: 'rgba(52,211,153,0.35)' }
      : { color: 'error.main', bg: 'rgba(248,113,113,0.10)', border: 'rgba(248,113,113,0.35)' }
    : waiting
      ? { color: 'warning.main', bg: 'rgba(251,191,36,0.10)', border: 'rgba(251,191,36,0.40)' }
      : { color: 'info.main', bg: 'rgba(56,189,248,0.08)', border: 'rgba(56,189,248,0.30)' };

  return (
    <Card
      sx={{
        p: 2,
        mb: 2,
        bgcolor: tone.bg,
        border: '1px solid',
        borderColor: tone.border,
      }}
    >
      <Stack
        direction={{ xs: 'column', md: 'row' }}
        spacing={2}
        sx={{ alignItems: { md: 'center' } }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, flexGrow: 1, minWidth: 0 }}>
          <StatusIcon status={status} waiting={waiting} busy={agentBusy} />
          <Box sx={{ minWidth: 0 }}>
            <Typography variant="subtitle1" sx={{ color: tone.color, fontWeight: 700 }}>
              {headline(status, haltReason, waiting)}
            </Typography>
            <Typography variant="body2" color="text.secondary" noWrap>
              {activity || (terminal ? `Halt reason: ${haltReason || 'n/a'}` : 'Working...')}
            </Typography>
          </Box>
        </Box>

        <Stack
          direction="row"
          spacing={1}
          sx={{ alignItems: 'center', flexShrink: 0, flexWrap: 'wrap' }}
        >
          <Tooltip title="Wall-clock time since you pressed trigger">
            <Chip size="small" variant="outlined" label={`${elapsed}s`} />
          </Tooltip>
          <Tooltip title="Loop iterations used against the ceiling">
            <Chip
              size="small"
              variant="outlined"
              label={`iter ${iteration}${maxIterations ? `/${maxIterations}` : ''}`}
            />
          </Tooltip>
          <Tooltip title="Spend so far against the per-incident cost ceiling">
            <Chip
              size="small"
              variant="outlined"
              label={`$${Number(totalCostUsd || 0).toFixed(4)}${budgetUsd ? ` / $${budgetUsd}` : ''}`}
            />
          </Tooltip>
          {!terminal && (
            <Chip
              size="small"
              variant="outlined"
              color={streamConnected ? 'success' : 'default'}
              label={streamConnected ? 'live' : 'connecting'}
            />
          )}
          {!terminal && (
            <Button
              size="small"
              color="inherit"
              variant="outlined"
              startIcon={<Stop />}
              disabled={cancelling}
              onClick={cancelIncident}
            >
              Cancel
            </Button>
          )}
        </Stack>
      </Stack>

      {!terminal && (
        <LinearProgress
          variant={agentBusy ? 'indeterminate' : 'determinate'}
          value={agentBusy ? undefined : 100}
          color={waiting ? 'warning' : 'info'}
          sx={{ height: 3, borderRadius: 3, mt: 1.5 }}
        />
      )}

      <StageStrip current={currentStage} terminal={terminal} />
    </Card>
  );
}

// A compact left-to-right map of the loop, so "where are we" is answerable at a
// glance without reading the timeline.
function StageStrip({ current, terminal }) {
  const idx = LOOP_STAGES.findIndex((s) => s.key === current);
  return (
    <Box sx={{ display: 'flex', gap: 0.5, mt: 1.5, flexWrap: 'wrap' }}>
      {LOOP_STAGES.map((stage, i) => {
        const done = idx >= 0 && i < idx;
        const active = i === idx && !terminal;
        return (
          <Tooltip key={stage.key} title={stage.blurb}>
            <Box
              sx={{
                px: 1,
                py: 0.25,
                borderRadius: 999,
                fontSize: 11,
                fontWeight: active ? 700 : 500,
                border: '1px solid',
                borderColor: active ? 'info.main' : 'divider',
                bgcolor: active ? 'rgba(56,189,248,0.18)' : 'transparent',
                color: active ? 'info.light' : done ? 'text.secondary' : 'text.disabled',
              }}
            >
              {stage.label}
            </Box>
          </Tooltip>
        );
      })}
    </Box>
  );
}

function StatusIcon({ status, waiting, busy }) {
  if (status === 'resolved') return <CheckCircle color="success" />;
  if (status === 'cancelled') return <Block color="error" />;
  if (status === 'escalated' || status === 'failed') return <ReportProblem color="error" />;
  if (waiting) return <Gavel color="warning" />;
  return <CircularProgress size={22} color={busy ? 'info' : 'inherit'} />;
}

function headline(status, haltReason, waiting) {
  if (waiting) return 'Your decision is needed';
  switch (status) {
    case 'resolved':
      return 'Resolved';
    case 'escalated':
      return 'Escalated to a human';
    case 'cancelled':
      return 'Cancelled by operator';
    case 'failed':
      return 'Failed';
    default:
      return 'Incident in progress';
  }
}

function IdleBar() {
  return (
    <Card sx={{ p: 2, mb: 2, border: '1px dashed', borderColor: 'divider' }}>
      <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
        No incident running
      </Typography>
      <Typography variant="body2" color="text.secondary">
        Pick a fault scenario on the left and press Trigger. A live run takes roughly 30-90 seconds
        and streams every decision here as it is made.
      </Typography>
    </Card>
  );
}
