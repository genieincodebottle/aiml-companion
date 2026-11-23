import { useEffect, useMemo, useState } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import {
  Alert,
  Box,
  Card,
  CardContent,
  Chip,
  Divider,
  FormControl,
  Grid,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Tooltip,
  Typography,
} from '@mui/material';
import { AccountTree } from '@mui/icons-material';
import { useIncidentStore } from '../../store/incidentStore';
import { incidentAPI } from '../../services/api';
import { useTheme } from '@mui/material/styles';
import { LOOP_STAGES, stageForEvent, agentMeta, statusColor } from '../Console/agentMeta';
import { toneFor } from '../../theme';
import AgentStepCard from '../Console/AgentStepCard';
import FlowDiagram from './FlowDiagram';
import LoopAnimation, { STAGE_TO_NODE } from './LoopAnimation';
import HowItWorks from './HowItWorks';

// Trace view: pick any incident, see the whole run mapped onto the loop, click a
// stage to read exactly what the agents decided there.
//
// The console answers "what is happening now". This answers "what happened, and
// where in the loop did it happen" - which is the question you actually have once
// a run is over, and the one a scrolling event list answers badly.
export default function WorkflowView() {
  const navigate = useNavigate();
  const mode = useTheme().palette.mode;
  const [params, setParams] = useSearchParams();
  const urlId = params.get('incident');

  const { incidentId: liveId, events: liveEvents, status: liveStatus } = useIncidentStore();

  const [list, setList] = useState([]);
  const [listError, setListError] = useState(null);
  // The user's pick; until they pick, follow the live incident (derived, not
  // copied into state by an effect).
  const [pickedId, setSelectedId] = useState(urlId || '');
  const selectedId = pickedId || liveId || '';
  const [loaded, setLoaded] = useState(null);
  const [stage, setStage] = useState(null);

  useEffect(() => {
    incidentAPI
      .listIncidents()
      .then((d) => setList(d.incidents || []))
      .catch((e) => setListError(e?.message || 'Could not list incidents'));
  }, [liveStatus]);

  // Fetch a stored incident whenever the selection is not the live one.
  useEffect(() => {
    let active = true;
    if (!selectedId || selectedId === liveId) return undefined;
    incidentAPI
      .getIncident(selectedId)
      .then((d) => active && setLoaded(d))
      .catch(() => active && setLoaded(null));
    return () => {
      active = false;
    };
  }, [selectedId, liveId]);

  const isLive = Boolean(selectedId) && selectedId === liveId;
  // A stored incident only counts once it is the one actually selected.
  const stored = !isLive && loaded?.incident_id === selectedId ? loaded : null;
  const events = useMemo(
    () => (isLive ? liveEvents : stored?.events || []),
    [isLive, liveEvents, stored]
  );
  const status = isLive ? liveStatus : stored?.status;

  // Bucket every event into the loop stage it belongs to.
  const byStage = useMemo(() => {
    const map = {};
    events.forEach((evt, idx) => {
      const key = stageForEvent(evt);
      if (!key) return;
      (map[key] = map[key] || []).push({ evt, idx });
    });
    return map;
  }, [events]);

  const counts = useMemo(
    () => Object.fromEntries(Object.entries(byStage).map(([k, v]) => [k, v.length])),
    [byStage]
  );

  const activeStage = useMemo(() => {
    if (!isLive) return null;
    for (let i = events.length - 1; i >= 0; i -= 1) {
      const s = stageForEvent(events[i]);
      if (s) return s;
    }
    return null;
  }, [events, isLive]);

  const shownStage = stage || activeStage || Object.keys(byStage)[0] || null;
  const stageMeta = LOOP_STAGES.find((s) => s.key === shownStage);
  const stageEvents = shownStage ? byStage[shownStage] || [] : [];

  return (
    <Box>
      {/* One row, not a card with a paragraph under it. The explanation of what
          this page is for moved onto the title, where it costs no vertical space
          until someone asks for it. */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 2, flexWrap: 'wrap' }}>
        <Tooltip title="Every step of a run, mapped onto the loop it came from. Click any stage to read what the agents actually decided there - the reasoning, the confidence, the objections, and the guardrails that fired.">
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, cursor: 'help' }}>
            <AccountTree color="primary" />
            <Typography variant="h6">Workflow trace</Typography>
          </Box>
        </Tooltip>
        {status && <Chip size="small" color={statusColor(status)} label={status} />}
        <Box sx={{ flexGrow: 1 }} />
        <FormControl size="small" sx={{ minWidth: 280 }}>
          <InputLabel id="incident-select">Incident</InputLabel>
          <Select
            labelId="incident-select"
            label="Incident"
            value={selectedId}
            onChange={(e) => {
              setSelectedId(e.target.value);
              setStage(null);
              setParams({ incident: e.target.value });
            }}
          >
            {list.length === 0 && (
              <MenuItem value="" disabled>
                No incidents yet
              </MenuItem>
            )}
            {list.map((it) => (
              <MenuItem key={it.incident_id} value={it.incident_id}>
                {it.scenario_id} - {it.status}
                {it.incident_id === liveId ? ' (live)' : ''}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>

      {listError && (
        <Alert severity="warning" variant="outlined" sx={{ mb: 2 }}>
          {listError}
        </Alert>
      )}

      {events.length === 0 ? (
        <HowItWorks onGoToConsole={() => navigate('/console')} />
      ) : (
        <Grid container spacing={2} sx={{ alignItems: 'flex-start' }}>
          {/* One card, not two stacked. The ring and the stage rail are two views
              of the same thing, so a heading between them was pure separation. */}
          <Grid size={{ xs: 12, md: 4.5 }}>
            <Card>
              <CardContent>
                <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
                  {isLive ? 'Where the loop is now' : 'Where this run got to'}
                </Typography>
                <LoopAnimation
                  height={280}
                  activeIndex={STAGE_TO_NODE[shownStage] ?? null}
                  interactive={false}
                />
                <Divider sx={{ my: 1.5 }} />
                <Tooltip title="Dimmed stages were never reached. That is information: a run that never hit Critique was vetoed nowhere, and one that never hit Act never touched the sandbox.">
                  <Typography
                    variant="caption"
                    color="text.secondary"
                    sx={{ display: 'block', mb: 1, cursor: 'help' }}
                  >
                    Every stage · dimmed means never reached
                  </Typography>
                </Tooltip>
                <FlowDiagram
                  counts={counts}
                  activeStage={activeStage}
                  selected={shownStage}
                  onSelect={setStage}
                />
              </CardContent>
            </Card>
          </Grid>

          <Grid size={{ xs: 12, md: 7.5 }}>
            <Card>
              <CardContent sx={{ pb: 1 }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
                  {stageMeta ? stageMeta.label : 'Select a stage'}
                </Typography>
                {stageMeta && (
                  <Stack direction="row" spacing={1} sx={{ mt: 0.5, mb: 0.5, flexWrap: 'wrap' }}>
                    {stageMeta.agent && (
                      <Chip
                        size="small"
                        variant="outlined"
                        label={stageMeta.agent}
                        sx={{ color: toneFor(agentMeta(stageMeta.agent).color, mode) }}
                      />
                    )}
                    <Chip
                      size="small"
                      variant="outlined"
                      label={`${stageEvents.length} event(s)`}
                    />
                  </Stack>
                )}
                {stageMeta && (
                  <Typography variant="body2" color="text.secondary">
                    {stageMeta.blurb}
                    {stageMeta.agent ? ` ${agentMeta(stageMeta.agent).does}` : ''}
                  </Typography>
                )}
              </CardContent>
              <Divider />
              <Box sx={{ p: 2, maxHeight: 'calc(100vh - 320px)', overflowY: 'auto' }}>
                {stageEvents.length === 0 ? (
                  <Typography variant="body2" color="text.secondary">
                    This stage was not reached in this run.
                  </Typography>
                ) : (
                  <Stack spacing={1.5}>
                    {stageEvents.map(({ evt, idx }) => (
                      <AgentStepCard key={idx} event={evt} />
                    ))}
                  </Stack>
                )}
              </Box>
            </Card>
          </Grid>
        </Grid>
      )}
    </Box>
  );
}
