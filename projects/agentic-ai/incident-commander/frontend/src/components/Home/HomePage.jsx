import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Alert,
  Box,
  Button,
  Card,
  CardActionArea,
  CardContent,
  Chip,
  Divider,
  Grid,
  LinearProgress,
  Stack,
  Tab,
  Tabs,
  Tooltip,
  Typography,
} from '@mui/material';
import {
  Bolt,
  AccountTree,
  History as HistoryIcon,
  Terminal,
  Shield,
  Gavel,
  VisibilityOff,
  Psychology,
  Speed,
  Savings,
  CheckCircle,
  ReportProblem,
} from '@mui/icons-material';
import { incidentAPI, scenarioAPI, memoryAPI, systemAPI } from '../../services/api';
import { useTheme } from '@mui/material/styles';
import { AGENT_META } from '../Console/agentMeta';
import { toneFor, tintFor, GRADIENTS } from '../../theme';
import LoopAnimation from '../Flow/LoopAnimation';

// The landing page.
//
// It has three jobs, in this order: tell a first-time visitor what this system is,
// show real numbers from this deployment rather than marketing claims, and route to
// the three things you can actually do.
//
// On density: an earlier version stacked four explanatory cards down the page, so
// understanding the system meant scrolling past roughly two screens of prose. The
// content was right and the shape was wrong. Everything below the fold now lives
// behind tabs - nothing was cut, it is just served one section at a time, which is
// how a page stays detailed without reading as a wall of text. Long explanations
// that are not the main point moved into tooltips, where they cost no space until
// someone wants them.

// The scenarios that make the evaluation meaningful get a colored label, so the
// catalog reads as "these are the interesting ones" rather than ten equal rows.
const SPECIAL = {
  cascading_failure: { label: 'compound · two ordered fixes', color: '#38bdf8' },
  poisoned_telemetry: { label: 'adversarial · injected text', color: '#fb923c' },
  silent_config_drift: { label: 'must request a tool', color: '#22d3ee' },
  unfixable_ca_expiry: { label: 'must escalate', color: '#f87171' },
  false_alarm: { label: 'must do nothing', color: '#f87171' },
};

export default function HomePage() {
  const navigate = useNavigate();
  const mode = useTheme().palette.mode;
  const [incidents, setIncidents] = useState([]);
  const [scenarios, setScenarios] = useState([]);
  const [memory, setMemory] = useState(null);
  const [config, setConfig] = useState(null);
  const [health, setHealth] = useState(null);
  //: Aggregates from the backend. With Postgres these span every run ever
  //: recorded; without it they match what is in this process's memory.
  const [serverStats, setServerStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState(0);

  useEffect(() => {
    let active = true;
    Promise.allSettled([
      incidentAPI.listIncidents(),
      scenarioAPI.getScenarios(),
      memoryAPI.getMemory(),
      incidentAPI.getConfig(),
      systemAPI.getHealth(),
      incidentAPI.getStats(),
    ]).then(([inc, scn, mem, cfg, hlt, st]) => {
      if (!active) return;
      if (inc.status === 'fulfilled') setIncidents(inc.value.incidents || []);
      if (scn.status === 'fulfilled') setScenarios(scn.value.scenarios || []);
      if (mem.status === 'fulfilled') setMemory(mem.value.stats || null);
      if (cfg.status === 'fulfilled') setConfig(cfg.value || null);
      if (hlt.status === 'fulfilled') setHealth(hlt.value || null);
      if (st.status === 'fulfilled') setServerStats(st.value || null);
      setLoading(false);
    });
    return () => {
      active = false;
    };
  }, []);

  // Every number below is computed from this deployment's own runs. A dashboard
  // that shows an aspirational figure is worse than one that shows nothing.
  const stats = useMemo(() => {
    const done = incidents.filter((i) => i.finalized);
    const resolved = done.filter((i) => i.status === 'resolved').length;
    const escalated = done.filter((i) => i.status === 'escalated').length;
    const spend = incidents.reduce((a, i) => a + (i.total_cost_usd || 0), 0);
    const iters = done.length ? done.reduce((a, i) => a + (i.iterations || 0), 0) / done.length : 0;
    // Server figures win when present: they are computed over the whole table
    // rather than over the page of incidents this request happened to return.
    return {
      total: serverStats?.total ?? incidents.length,
      done: done.length,
      resolved: serverStats?.resolved ?? resolved,
      escalated: serverStats?.escalated ?? escalated,
      spend: serverStats?.total_cost_usd ?? spend,
      meanIterations: serverStats?.mean_iterations ?? iters,
      running: incidents.filter((i) => !i.finalized).length,
    };
  }, [incidents, serverStats]);

  // Incidents are refused only when Gemini was forced (IC_LLM_MODE=gemini) with no key.
  const llmBlocked = health?.status === 'degraded';
  const settled = stats.resolved + stats.escalated;

  return (
    <Box>
      {/* ---------------- hero ---------------- */}
      <Card sx={{ mb: 2, overflow: 'hidden' }}>
        <Box sx={{ p: { xs: 2.5, md: 4 }, background: GRADIENTS.hero(mode) }}>
          <Grid container spacing={3} sx={{ alignItems: 'center' }}>
            <Grid size={{ xs: 12, md: 6 }}>
              <Chip
                size="small"
                color="primary"
                variant="outlined"
                label="Multi-agent systems · loop engineering"
              />
              <Typography variant="h4" sx={{ mt: 1.5, mb: 1, fontWeight: 700, lineHeight: 1.2 }}>
                An autonomous SRE
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
                Seven agents diagnose and fix a production incident. They argue with each other,
                stop for your approval, and escalate rather than guess.
              </Typography>

              <Tooltip title="The LLM routes, diagnoses, proposes, critiques and verifies. It never sets a ceiling, grants a permission, classifies its own risk, or waives an approval. Telemetry is attacker-influenced text that reaches every prompt, so anything a model can influence must not be able to widen the model's own permissions.">
                <Box
                  sx={{
                    p: 1.5,
                    mb: 2.5,
                    borderRadius: 2,
                    borderLeft: '3px solid',
                    borderColor: 'primary.main',
                    bgcolor: 'action.hover',
                    cursor: 'help',
                  }}
                >
                  <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
                    The model decides what to do. Code decides what is allowed.
                  </Typography>
                </Box>
              </Tooltip>

              <Stack direction="row" spacing={1.5} sx={{ flexWrap: 'wrap', gap: 1.5 }}>
                <Button
                  variant="contained"
                  size="large"
                  startIcon={<Bolt />}
                  onClick={() => navigate('/console')}
                  sx={{
                    px: 3,
                    background: GRADIENTS.brand,
                    '&:hover': { background: GRADIENTS.brandHover },
                  }}
                >
                  Run an incident
                </Button>
                <Button
                  variant="outlined"
                  size="large"
                  startIcon={<AccountTree />}
                  onClick={() => navigate('/flow')}
                >
                  See how it works
                </Button>
              </Stack>
            </Grid>

            {/* The one thing on this page that moves. The loop is the product, and
                a static list of agent names does not convey that it goes round. */}
            <Grid size={{ xs: 12, md: 6 }}>
              <LoopAnimation height={320} />
            </Grid>
          </Grid>

          {!loading && llmBlocked && (
            <Alert severity="warning" variant="outlined" sx={{ mt: 2 }}>
              <code>IC_LLM_MODE=gemini</code> is set but no Gemini API key is configured, so the
              backend refuses to start incidents. Put <code>GOOGLE_API_KEY</code> in{' '}
              <code>backend/.env</code> and restart, or unset <code>IC_LLM_MODE</code> to use
              offline mode.
            </Alert>
          )}
        </Box>
      </Card>

      {loading && <LinearProgress sx={{ mb: 2 }} />}

      {/* ---------------- live numbers ---------------- */}
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Stat
          xs={6}
          md={3}
          icon={<Terminal fontSize="small" />}
          label="Incidents run here"
          value={stats.total}
          sub={stats.running ? `${stats.running} in progress` : `${stats.done} finished`}
        />
        <Stat
          xs={6}
          md={3}
          icon={<CheckCircle fontSize="small" />}
          color="success"
          label="Resolved"
          value={settled ? `${Math.round((stats.resolved / settled) * 100)}%` : '-'}
          sub={`${stats.resolved} resolved, ${stats.escalated} escalated`}
          tip="Escalating is the correct outcome for two of the ten scenarios, so this is not a score to maximise."
        />
        <Stat
          xs={6}
          md={3}
          icon={<Savings fontSize="small" />}
          color="warning"
          label="Spend on this machine"
          value={`$${stats.spend.toFixed(4)}`}
          sub={config ? `ceiling $${config.max_cost_usd}/incident` : ''}
          tip="Cost is enforced twice: at the top of each pass, and pre-flight before every single model call."
        />
        <Stat
          xs={6}
          md={3}
          icon={<Speed fontSize="small" />}
          color="info"
          label="Mean iterations"
          value={stats.meanIterations ? stats.meanIterations.toFixed(2) : '-'}
          sub={config ? `ceiling ${config.max_iterations}` : ''}
          tip="How many times round the loop, on average, before it stopped."
        />
      </Grid>

      {/* ---------------- where to go: one compact row ---------------- */}
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <NavCard
          icon={<Terminal />}
          color="#4f9eff"
          title="Console"
          blurb="Trigger a fault, watch it live."
          detail="Every decision, confidence score and objection streams in as it is made, and risky fixes stop and wait for you."
          onClick={() => navigate('/console')}
        />
        <NavCard
          icon={<AccountTree />}
          color="#22d3ee"
          title="Workflow trace"
          blurb="Replay any run, stage by stage."
          detail="Click a stage to read what the agents actually decided there - the reasoning, the evidence, the vetoes, and the guardrails that fired."
          onClick={() => navigate('/flow')}
        />
        <NavCard
          icon={<HistoryIcon />}
          color="#34d399"
          title="History"
          blurb="Past incidents and postmortems."
          detail="Halt reasons and per-agent cost. Run the same scenario twice and the second run recalls the first from memory."
          onClick={() => navigate('/history')}
        />
      </Grid>

      {/* ---------------- the detail, one section at a time ---------------- */}
      <Card>
        <Tabs
          value={tab}
          onChange={(_, v) => setTab(v)}
          variant="scrollable"
          scrollButtons="auto"
          sx={{ px: 1, borderBottom: '1px solid', borderColor: 'divider' }}
        >
          <Tab label="The agents" />
          <Tab label="Why it is not a chatbot" />
          <Tab label={`Faults (${scenarios.length || 10})`} />
          <Tab label="Runtime" />
          <Tab label="Recent runs" />
        </Tabs>

        <CardContent>
          {tab === 0 && (
            <Grid container spacing={1.5}>
              {Object.entries(AGENT_META)
                .filter(([name]) => name !== 'Orchestrator')
                .map(([name, meta]) => (
                  <Grid size={{ xs: 12, sm: 6, md: 4 }} key={name}>
                    <Box sx={{ display: 'flex', gap: 1.25, alignItems: 'flex-start' }}>
                      <Box
                        sx={{
                          width: 28,
                          height: 28,
                          borderRadius: '50%',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          bgcolor: tintFor(meta.color, mode),
                          color: toneFor(meta.color, mode),
                          flexShrink: 0,
                          mt: 0.2,
                        }}
                      >
                        {meta.icon}
                      </Box>
                      <Box>
                        <Typography variant="subtitle2" sx={{ color: toneFor(meta.color, mode) }}>
                          {name}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {meta.does}
                        </Typography>
                      </Box>
                    </Box>
                  </Grid>
                ))}
            </Grid>
          )}

          {tab === 1 && (
            <Grid container spacing={2}>
              <Feature
                icon={<Gavel fontSize="small" />}
                title="Six hard halt conditions"
                body="Iterations, cost, wall-clock, human rejection, approval timeout, operator cancel. Each has a test that drives it to a real stop."
              />
              <Feature
                icon={<Shield fontSize="small" />}
                title="Injection quarantined, not flagged"
                body="Attacker text is removed and fenced before any agent sees it. The architecture is the defence, not the regex list."
              />
              <Feature
                icon={<VisibilityOff fontSize="small" />}
                title="Willing to return nothing"
                body="Two scenarios are scored on the agent NOT acting. An agent evaluated only on faults it can fix will invent a fix for one it cannot."
              />
              <Feature
                icon={<Psychology fontSize="small" />}
                title="Uncertainty wired to behaviour"
                body="Low confidence makes the loop fetch more telemetry and re-diagnose, then escalate if it still cannot commit. It is not just displayed."
              />
            </Grid>
          )}

          {tab === 2 && (
            <>
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ display: 'block', mb: 1.5 }}
              >
                The agent never sees the answer - it infers it from telemetry. Hover a card for the
                brief.
              </Typography>
              <Grid container spacing={1}>
                {scenarios.map((s) => (
                  <Grid size={{ xs: 6, sm: 4, md: 3 }} key={s.id}>
                    <Tooltip title={s.description || ''}>
                      <Box
                        onClick={() => navigate('/console')}
                        sx={{
                          p: 1,
                          height: '100%',
                          borderRadius: 1.5,
                          cursor: 'pointer',
                          border: '1px solid',
                          borderColor: SPECIAL[s.id]
                            ? toneFor(SPECIAL[s.id].color, mode)
                            : 'divider',
                          '&:hover': { borderColor: 'primary.main' },
                        }}
                      >
                        <Typography variant="body2" sx={{ fontWeight: 600, lineHeight: 1.3 }}>
                          {s.name}
                        </Typography>
                        <Typography
                          variant="caption"
                          sx={{
                            color: SPECIAL[s.id]
                              ? toneFor(SPECIAL[s.id].color, mode)
                              : 'text.secondary',
                          }}
                        >
                          {SPECIAL[s.id] ? SPECIAL[s.id].label : s.severity || 'fault'}
                        </Typography>
                      </Box>
                    </Tooltip>
                  </Grid>
                ))}
                {!scenarios.length && !loading && (
                  <Grid size={{ xs: 12 }}>
                    <Typography variant="body2" color="text.secondary">
                      Could not reach the backend to list scenarios.
                    </Typography>
                  </Grid>
                )}
              </Grid>
            </>
          )}

          {tab === 3 && (
            <>
              <Grid container spacing={1} sx={{ mb: 1.5 }}>
                <KV k="Model" v={health?.model || '-'} />
                <KV k="Fast tier" v={health?.model_fast || 'off'} />
                <KV k="Engine" v={health?.engine || '-'} />
                <KV k="Agent role" v={health?.role || '-'} />
                {config && <KV k="Confidence floor" v={config.min_action_confidence} />}
                {config && <KV k="Evidence rounds" v={config.max_evidence_rounds} />}
                {config && <KV k="Iteration ceiling" v={config.max_iterations} />}
                {config && <KV k="Cost ceiling" v={`$${config.max_cost_usd}`} />}
                {memory && <KV k="Episodes in memory" v={memory.episodes ?? 0} />}
                {memory && <KV k="Scenarios seen" v={(memory.scenarios_seen || []).length} />}
                <KV k="Persistence" v={health?.db_connected ? 'postgres' : 'in-memory'} />
                <KV k="Login" v={health?.login_required ? 'required' : 'off'} />
              </Grid>
              <Typography variant="caption" color="text.secondary">
                Every limit here is served from <code>/api/config</code>, so the agent&apos;s real
                constraints are inspectable at runtime rather than only documented.
              </Typography>
            </>
          )}

          {tab === 4 &&
            (incidents.length ? (
              <Stack spacing={1} divider={<Divider flexItem />}>
                {incidents.slice(0, 8).map((i) => (
                  <Box
                    key={i.incident_id}
                    sx={{ display: 'flex', alignItems: 'center', gap: 1, cursor: 'pointer' }}
                    onClick={() => navigate(`/flow?incident=${i.incident_id}`)}
                  >
                    {i.status === 'resolved' ? (
                      <CheckCircle fontSize="small" color="success" />
                    ) : (
                      <ReportProblem fontSize="small" color="error" />
                    )}
                    <Typography variant="body2" sx={{ flexGrow: 1, minWidth: 0 }} noWrap>
                      {i.scenario_id}
                    </Typography>
                    <Chip size="small" variant="outlined" label={i.halt_reason || i.status} />
                    <Typography variant="caption" color="text.secondary">
                      ${Number(i.total_cost_usd || 0).toFixed(4)}
                    </Typography>
                  </Box>
                ))}
              </Stack>
            ) : (
              <Typography variant="body2" color="text.secondary">
                Nothing has run yet. Trigger an incident from the console and it will appear here.
              </Typography>
            ))}
        </CardContent>
      </Card>
    </Box>
  );
}

// Icon and number on one line, label beneath. The number is the thing being
// communicated, so it gets the size - a stat card whose label outweighs its value
// is a caption with extra steps.
function Stat({ xs, md, icon, label, value, sub, tip, color = 'primary' }) {
  const inner = (
    <Card
      sx={{ height: '100%', transition: 'box-shadow 200ms ease-out', '&:hover': { boxShadow: 4 } }}
    >
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
          <Box sx={{ color: `${color}.main`, display: 'flex' }}>{icon}</Box>
          <Typography variant="h4" sx={{ fontWeight: 800, lineHeight: 1.1 }}>
            {value}
          </Typography>
        </Box>
        <Typography variant="body2" color="text.secondary">
          {label}
        </Typography>
        {sub && (
          <Typography variant="caption" color="text.disabled" sx={{ display: 'block', mt: 0.25 }}>
            {sub}
          </Typography>
        )}
      </CardContent>
    </Card>
  );
  return (
    <Grid size={{ xs: xs, md: md }}>{tip ? <Tooltip title={tip}>{inner}</Tooltip> : inner}</Grid>
  );
}

// One line on the card, the full explanation on hover. These used to be
// three-line paragraphs stacked down the left column, which is most of what made
// the page feel like an essay.
function NavCard({ icon, color, title, blurb, detail, onClick }) {
  const mode = useTheme().palette.mode;
  const accent = toneFor(color, mode);
  return (
    <Grid size={{ xs: 12, md: 4 }}>
      <Tooltip title={detail}>
        <Card
          sx={{
            height: '100%',
            transition: 'box-shadow 200ms ease-out',
            '&:hover': { boxShadow: 4 },
          }}
        >
          <CardActionArea onClick={onClick} sx={{ height: '100%' }}>
            <CardContent sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
              <Box
                sx={{
                  width: 38,
                  height: 38,
                  borderRadius: 2,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  bgcolor: tintFor(color, mode),
                  color: accent,
                  flexShrink: 0,
                }}
              >
                {icon}
              </Box>
              <Box sx={{ minWidth: 0 }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 700, lineHeight: 1.2 }}>
                  {title}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {blurb}
                </Typography>
              </Box>
            </CardContent>
          </CardActionArea>
        </Card>
      </Tooltip>
    </Grid>
  );
}

function Feature({ icon, title, body }) {
  return (
    <Grid size={{ xs: 12, sm: 6 }}>
      <Box sx={{ display: 'flex', gap: 1.25, alignItems: 'flex-start' }}>
        <Box sx={{ color: 'info.main', mt: 0.3 }}>{icon}</Box>
        <Box>
          <Typography variant="subtitle2">{title}</Typography>
          <Typography variant="caption" color="text.secondary">
            {body}
          </Typography>
        </Box>
      </Box>
    </Grid>
  );
}

function KV({ k, v }) {
  return (
    <Grid size={{ xs: 12, sm: 6, md: 4 }}>
      <Box sx={{ display: 'flex', gap: 1, alignItems: 'baseline' }}>
        <Typography variant="body2" color="text.secondary" sx={{ flexGrow: 1 }}>
          {k}
        </Typography>
        <Typography variant="body2" sx={{ fontFamily: 'ui-monospace, monospace', fontWeight: 600 }}>
          {String(v)}
        </Typography>
      </Box>
    </Grid>
  );
}
