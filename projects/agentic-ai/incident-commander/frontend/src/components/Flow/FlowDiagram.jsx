import { Box, Chip, Tooltip, Typography } from '@mui/material';
import { ArrowUpward } from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import { LOOP_STAGES, agentMeta } from '../Console/agentMeta';
import { toneFor, tintFor } from '../../theme';

// The loop as a clickable rail of stages.
//
// Deliberately hand-rolled rather than a graph library: the control flow is one
// loop with three back-edges, and a force-directed layout expresses that less
// clearly than a straight line with the back-edges named underneath. It also
// keeps a dependency out of the bundle whose only job would be drawing ten boxes.
//
// On density: every row used to carry its full blurb, every gap carried a labelled
// arrow, and the back-edge notes printed the same sentence they already had as a
// tooltip. That is three copies of the same explanation stacked vertically. Rows
// are now one line each and the prose lives on hover, where it costs nothing until
// someone wants it.
export default function FlowDiagram({ counts = {}, activeStage, selected, onSelect }) {
  const mode = useTheme().palette.mode;

  return (
    <Box>
      <Box sx={{ position: 'relative' }}>
        {/* One continuous rail behind the nodes, so the stages read as a single
            thread rather than ten unrelated rows. */}
        <Box
          sx={{
            position: 'absolute',
            left: 15,
            top: 14,
            bottom: 14,
            width: 2,
            bgcolor: 'divider',
            borderRadius: 2,
          }}
        />

        {LOOP_STAGES.map((stage, i) => {
          const n = counts[stage.key] || 0;
          const visited = n > 0;
          const isActive = stage.key === activeStage;
          const isSelected = stage.key === selected;
          const am = stage.agent ? agentMeta(stage.agent) : null;
          const raw = am ? am.color : '#94a3b8';
          const accent = toneFor(raw, mode);
          const accentTint = tintFor(raw, mode);

          return (
            <Tooltip
              key={stage.key}
              placement="right"
              title={
                <>
                  <b>{stage.label}</b>
                  {stage.agent ? ` · ${stage.agent}` : ''}
                  <br />
                  {stage.blurb}
                  {edgeLabel(stage.key) ? (
                    <>
                      <br />→ {edgeLabel(stage.key)}
                    </>
                  ) : null}
                </>
              }
            >
              <Box
                onClick={() => onSelect?.(stage.key)}
                sx={{
                  position: 'relative',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1.25,
                  px: 1,
                  py: 0.6,
                  mb: 0.25,
                  borderRadius: 2,
                  cursor: 'pointer',
                  bgcolor: isSelected ? accentTint : 'transparent',
                  opacity: visited || isActive || isSelected ? 1 : 0.45,
                  transition: 'background-color 150ms ease-out',
                  '&:hover': { bgcolor: isSelected ? accentTint : 'action.hover' },
                }}
              >
                <Box
                  sx={{
                    width: 30,
                    height: 30,
                    borderRadius: '50%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0,
                    // Opaque, so the rail passes behind the node rather than through it.
                    bgcolor: visited ? accent : 'background.paper',
                    color: visited ? '#fff' : accent,
                    border: '2px solid',
                    borderColor: isActive ? 'info.main' : accent,
                    boxShadow: isActive ? 3 : 0,
                  }}
                >
                  {am ? am.icon : <Typography variant="caption">{i + 1}</Typography>}
                </Box>

                <Typography
                  variant="body2"
                  sx={{
                    flexGrow: 1,
                    minWidth: 0,
                    fontWeight: isSelected || isActive ? 700 : 500,
                    color: visited ? 'text.primary' : 'text.secondary',
                  }}
                  noWrap
                >
                  {stage.label}
                </Typography>

                {isActive && <Chip size="small" color="info" label="now" sx={CHIP} />}
                {n > 0 && (
                  <Chip
                    size="small"
                    label={n}
                    sx={{ ...CHIP, bgcolor: accentTint, color: accent }}
                  />
                )}
              </Box>
            </Tooltip>
          );
        })}
      </Box>

      <Box sx={{ mt: 1.5, display: 'flex', gap: 0.75, flexWrap: 'wrap', alignItems: 'center' }}>
        <ArrowUpward sx={{ fontSize: 14, color: 'warning.main' }} />
        <Typography variant="caption" color="text.secondary" sx={{ mr: 0.5 }}>
          Back-edges:
        </Typography>
        {BACK_EDGES.map((e) => (
          <Tooltip key={e.label} title={e.why}>
            <Chip
              size="small"
              variant="outlined"
              label={e.label}
              sx={{ ...CHIP, cursor: 'help' }}
            />
          </Tooltip>
        ))}
      </Box>
    </Box>
  );
}

//: Compact chip metrics, reused so the row height stays constant.
const CHIP = { height: 20, fontSize: '0.7rem', fontWeight: 700 };

// The three ways the loop goes backwards. Each is bounded - that bound is the
// whole reason an agent loop terminates, so it belongs in the tooltip.
const BACK_EDGES = [
  {
    label: 'verify → route',
    why: 'Not recovered, but the Verifier sees a productive next step. Bounded by the iteration ceiling.',
  },
  {
    label: 'critique → diagnose',
    why: 'The Critic vetoed the fix. Bounded by IC_MAX_CRITIC_VETOES, so the planner and reviewer cannot argue forever.',
  },
  {
    label: 'diagnose → diagnose',
    why: 'needs_more_data: run the requested read-only tools, then diagnose again. Bounded by IC_MAX_EVIDENCE_ROUNDS.',
  },
];

function edgeLabel(key) {
  switch (key) {
    case 'observe':
      return 'sanitized telemetry';
    case 'route':
      return 'or escalate, if no agent can help';
    case 'diagnose':
      return 'only if confidence clears the floor';
    case 'critique':
      return 'if endorsed';
    case 'approve':
      return 'only on an approved action';
    case 'verify':
      return 'if recovered';
    default:
      return '';
  }
}
