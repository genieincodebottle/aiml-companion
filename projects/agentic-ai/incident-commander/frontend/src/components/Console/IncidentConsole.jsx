import { useEffect, useMemo } from 'react';
import { Box, Grid, Alert, Card, CardContent, Collapse, Stack, Button } from '@mui/material';
import { AccountTree } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useIncidentStore } from '../../store/incidentStore';
import { stageForEvent } from './agentMeta';
import LoopAnimation, { STAGE_TO_NODE } from '../Flow/LoopAnimation';
import FaultInjectorPanel from './FaultInjectorPanel';
import LoopTimeline from './LoopTimeline';
import ApprovalGate from './ApprovalGate';
import LiveMetrics from './LiveMetrics';
import PostmortemReport from './PostmortemReport';
import RunStatusBar from './RunStatusBar';

export default function IncidentConsole() {
  const navigate = useNavigate();
  const {
    events,
    streamConnected,
    postmortem,
    error,
    incidentId,
    disconnectStream,
    reconnectStream,
    loadRuntimeConfig,
  } = useIncidentStore();

  // Close the SSE stream when leaving the console, and pick a still-running
  // incident back up on return - the store resumes from the last seen seq, so
  // nothing is replayed. Without the reconnect, navigating away mid-run left the
  // console frozen on "connecting" forever.
  useEffect(() => {
    reconnectStream();
    return () => disconnectStream();
  }, [reconnectStream, disconnectStream]);

  // The ceilings the loop is actually running under, for the metrics panel.
  useEffect(() => {
    loadRuntimeConfig();
  }, [loadRuntimeConfig]);

  // Which stage of the loop the latest event belongs to, so the ring can track it.
  const currentStage = useMemo(() => {
    for (let i = events.length - 1; i >= 0; i -= 1) {
      const s = stageForEvent(events[i]);
      if (s) return s;
    }
    return null;
  }, [events]);

  return (
    <Box>
      <Collapse in={Boolean(error)}>
        <Alert severity="error" variant="outlined" sx={{ mb: 2 }}>
          {error}
        </Alert>
      </Collapse>

      {/* Always the first thing on the page: is anything happening, and what. */}
      <RunStatusBar />

      <Grid container spacing={2} sx={{ alignItems: 'stretch' }}>
        {/* Left: fault injector.
            No fixed height. It used to inherit the timeline's
            `calc(100vh - 280px)`, and once the header and the footer controls took
            their share the scrollable middle collapsed to about one line - so the
            scenario name was clipped mid-word and its description was invisible
            behind a scrollbar. It now sizes to its content. */}
        <Grid size={{ xs: 12, md: 4 }}>
          <FaultInjectorPanel />
        </Grid>

        {/* Center: live loop timeline. This one does want a viewport-locked height,
            because it is the thing that scrolls as events stream in. */}
        <Grid size={{ xs: 12, md: 4.5 }}>
          <Box sx={{ height: { md: 'calc(100vh - 280px)' } }}>
            <LoopTimeline events={events} streamConnected={streamConnected} />
          </Box>
        </Grid>

        {/* Right: the decision you may need to make, then the numbers */}
        <Grid size={{ xs: 12, md: 3.5 }}>
          <Stack spacing={2}>
            {incidentId && (
              <Card>
                <CardContent sx={{ pb: 1 }}>
                  <LoopAnimation
                    height={250}
                    activeIndex={STAGE_TO_NODE[currentStage] ?? null}
                    interactive={false}
                  />
                </CardContent>
              </Card>
            )}
            <ApprovalGate />
            <LiveMetrics />
            {incidentId && (
              <Button
                variant="outlined"
                startIcon={<AccountTree />}
                onClick={() => navigate(`/flow?incident=${incidentId}`)}
              >
                Trace this run stage by stage
              </Button>
            )}
          </Stack>
        </Grid>
      </Grid>

      {/* Postmortem appears full width once the loop resolves */}
      <Collapse in={Boolean(postmortem)}>
        <Box sx={{ mt: 2 }}>
          <PostmortemReport postmortem={postmortem} />
        </Box>
      </Collapse>
    </Box>
  );
}
