import { useEffect, useMemo, useRef, useState } from 'react';
import { Box, Chip, IconButton, Tooltip, Typography } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { PlayArrow, Pause, Replay } from '@mui/icons-material';
import { AGENT_META } from '../Console/agentMeta';
import { toneFor, tintFor } from '../../theme';

// An animated ring of the seven agents, with a packet travelling the loop.
//
// Hand-drawn SVG rather than a chart library on purpose: the shape being taught
// is a *cycle with a gate in it*, and no off-the-shelf diagram component draws
// that. It is also the only thing on the landing page that moves, which is the
// point - the loop is the product, and a static list of agent names does not
// convey that it goes round.
//
// Two modes:
//   * `live` - driven by a real run's current stage (no timer)
//   * demo   - a timer walks the ring so the page is not inert before you run one

const NODES = [
  { key: 'Triage', label: 'Triage', short: 'severity' },
  { key: 'Diagnostician', label: 'Diagnose', short: 'root cause' },
  { key: 'Remediation', label: 'Propose', short: 'one fix' },
  { key: 'Critic', label: 'Critique', short: 'argue back' },
  { key: 'gate', label: 'Human gate', short: 'you decide' },
  { key: 'act', label: 'Act', short: 'sandbox' },
  { key: 'Verifier', label: 'Verify', short: 'did it work?' },
  { key: 'Postmortem', label: 'Postmortem', short: 'store lesson' },
];

const GATE_COLOR = '#fbbf24';
const ACT_COLOR = '#f87171';

function baseColor(key) {
  if (key === 'gate') return GATE_COLOR;
  if (key === 'act') return ACT_COLOR;
  return AGENT_META[key]?.color || '#94a3b8';
}

export default function LoopAnimation({
  activeIndex = null,
  height = 340,
  interactive = true,
  onBrand = false,
}) {
  const isLive = activeIndex != null;
  const theme = useTheme();
  const mode = theme.palette.mode;

  // The SVG cannot inherit MUI colors, so every fill it uses is resolved here.
  // These were hardcoded for dark, which made the centre label invisible on white.
  // `onBrand` draws the diagram on the login page's coloured panel, where the
  // theme's text colours would be invisible and the accents would fight the
  // gradient. Everything goes white there and the hues are dropped.
  const ink = onBrand ? '#f8fafc' : theme.palette.text.primary;
  const muted = onBrand ? 'rgba(248,250,252,0.75)' : theme.palette.text.secondary;
  const nodeFill = onBrand ? 'rgba(255,255,255,0.10)' : theme.palette.background.paper;
  const onAccent = onBrand ? '#0b1220' : mode === 'light' ? '#ffffff' : '#0b0f17';
  const ring = onBrand
    ? 'rgba(255,255,255,0.30)'
    : mode === 'light'
      ? 'rgba(71,85,105,0.28)'
      : 'rgba(148,163,184,0.22)';
  const dash = onBrand
    ? 'rgba(255,255,255,0.75)'
    : mode === 'light'
      ? 'rgba(2,132,199,0.55)'
      : 'rgba(56,189,248,0.55)';
  const nodeColor = (key) => (onBrand ? '#f8fafc' : toneFor(baseColor(key), mode));
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [hover, setHover] = useState(null);
  const timer = useRef(null);

  useEffect(() => {
    if (isLive || !playing) return undefined;
    timer.current = setInterval(() => setStep((s) => (s + 1) % NODES.length), 1400);
    return () => clearInterval(timer.current);
  }, [isLive, playing]);

  const current = isLive ? activeIndex : step;

  // Lay the nodes out on a circle. Angles start at the top and go clockwise so the
  // reading order matches the numbered order.
  const geometry = useMemo(() => {
    const cx = 200;
    const cy = 175;
    const r = 128;
    return NODES.map((n, i) => {
      const angle = (i / NODES.length) * Math.PI * 2 - Math.PI / 2;
      return { ...n, i, x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle), angle };
    });
  }, []);

  const shown = hover != null ? hover : current;
  const shownNode = NODES[shown];

  return (
    <Box>
      <Box sx={{ position: 'relative' }}>
        <svg viewBox="0 0 400 366" style={{ width: '100%', height, display: 'block' }}>
          <defs>
            <marker id="arrow" markerWidth="7" markerHeight="7" refX="5" refY="2.5" orient="auto">
              <path d="M0,0 L5,2.5 L0,5 z" fill="rgba(148,163,184,0.55)" />
            </marker>
            <radialGradient id="coreGlow">
              <stop
                offset="0%"
                stopColor={mode === 'light' ? 'rgba(29,111,212,0.16)' : 'rgba(79,158,255,0.30)'}
              />
              <stop offset="100%" stopColor="rgba(79,158,255,0)" />
            </radialGradient>
          </defs>

          {/* the loop itself */}
          <circle cx="200" cy="175" r="128" fill="none" stroke={ring} strokeWidth="1.5" />
          <circle
            cx="200"
            cy="175"
            r="128"
            fill="none"
            stroke={dash}
            strokeWidth="2"
            strokeDasharray="6 10"
            strokeLinecap="round"
          >
            <animate
              attributeName="stroke-dashoffset"
              from="80"
              to="0"
              dur="2.4s"
              repeatCount="indefinite"
            />
          </circle>

          {/* centre */}
          <circle cx="200" cy="175" r="62" fill="url(#coreGlow)" />
          <text x="200" y="167" textAnchor="middle" fontSize="13" fontWeight="700" fill={ink}>
            Agent loop
          </text>
          <text x="200" y="184" textAnchor="middle" fontSize="9.5" fill={muted}>
            bounded · supervised
          </text>
          <text x="200" y="198" textAnchor="middle" fontSize="9.5" fill={muted}>
            halts on 6 conditions
          </text>

          {geometry.map((n) => {
            const active = n.i === shown;
            const color = nodeColor(n.key);
            const passed = !isLive && n.i < current;
            return (
              <g
                key={n.key}
                onMouseEnter={() => interactive && setHover(n.i)}
                onMouseLeave={() => interactive && setHover(null)}
                style={{ cursor: interactive ? 'pointer' : 'default' }}
              >
                {active && (
                  <circle cx={n.x} cy={n.y} r="26" fill={color} opacity="0.18">
                    <animate
                      attributeName="r"
                      values="22;30;22"
                      dur="1.6s"
                      repeatCount="indefinite"
                    />
                    <animate
                      attributeName="opacity"
                      values="0.28;0.05;0.28"
                      dur="1.6s"
                      repeatCount="indefinite"
                    />
                  </circle>
                )}
                <circle
                  cx={n.x}
                  cy={n.y}
                  r="19"
                  fill={active ? color : nodeFill}
                  stroke={color}
                  strokeWidth={active ? 2.5 : 1.5}
                  opacity={active || passed ? 1 : 0.75}
                />
                <text
                  x={n.x}
                  y={n.y + 4}
                  textAnchor="middle"
                  fontSize="11"
                  fontWeight="700"
                  fill={active ? onAccent : color}
                >
                  {n.i + 1}
                </text>
                <text
                  x={n.x}
                  y={n.y + (n.y < 175 ? -27 : 34)}
                  textAnchor="middle"
                  fontSize="10.5"
                  fontWeight={active ? 700 : 500}
                  fill={active ? ink : muted}
                >
                  {n.label}
                </text>
              </g>
            );
          })}

          {/* the human gate is the one place the loop is allowed to stop dead.
              y=358, below the bottom node's own label at y=337 - at 332 the two
              strings overlapped and both became unreadable. */}
          <text x="200" y="358" textAnchor="middle" fontSize="9.5" fill={nodeColor('gate')}>
            ● step 5 blocks until a human answers
          </text>
        </svg>

        {interactive && !isLive && (
          <Box sx={{ position: 'absolute', top: 0, right: 0, display: 'flex', gap: 0.5 }}>
            <Tooltip title={playing ? 'Pause' : 'Play'}>
              <IconButton size="small" onClick={() => setPlaying((p) => !p)}>
                {playing ? <Pause fontSize="small" /> : <PlayArrow fontSize="small" />}
              </IconButton>
            </Tooltip>
            <Tooltip title="Restart">
              <IconButton size="small" onClick={() => setStep(0)}>
                <Replay fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>
        )}
      </Box>

      {/* caption for whichever node is lit */}
      <Box
        sx={{
          mt: 1,
          p: 1.25,
          minHeight: 62,
          borderRadius: 2,
          border: '1px solid',
          borderColor: 'divider',
          bgcolor: 'rgba(148,163,184,0.05)',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.25 }}>
          <Chip
            size="small"
            label={`${shown + 1}. ${shownNode.label}`}
            sx={{
              bgcolor: tintFor(baseColor(shownNode.key), mode),
              color: nodeColor(shownNode.key),
              fontWeight: 700,
            }}
          />
          <Typography variant="caption" color="text.secondary">
            {shownNode.short}
          </Typography>
        </Box>
        <Typography variant="body2" color="text.secondary">
          {CAPTIONS[shownNode.key]}
        </Typography>
      </Box>
    </Box>
  );
}

const CAPTIONS = {
  Triage: 'Sets severity and blast radius from the first look at telemetry. Runs once.',
  Diagnostician:
    'Ranks root-cause hypotheses with a confidence score. Below the floor it asks for more telemetry instead of guessing.',
  Remediation:
    'Picks exactly one action from a closed menu and predicts a specific signal that must change if it is right.',
  Critic:
    'A second agent told to find the flaw. A veto sends the loop back to re-diagnose, not to a human.',
  gate: 'High-risk actions always stop here. The loop is genuinely blocked - and escalates if nobody answers in time.',
  act: 'The single mutating call in the whole system. Everything else is read-only by construction.',
  Verifier:
    'Reads fresh telemetry and checks the prediction. Recovery the prediction did not explain is flagged.',
  Postmortem:
    'Written for every incident, including failures, and stored as a lesson the next run can retrieve.',
};

// Map a loop-stage key to this ring's index, so a live run can drive it.
export const STAGE_TO_NODE = {
  triage: 0,
  diagnose: 1,
  propose: 2,
  critique: 3,
  approve: 4,
  act: 5,
  verify: 6,
  finish: 7,
};
