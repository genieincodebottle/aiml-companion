import { useEffect, useMemo, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Divider,
  Stack,
  Typography,
} from '@mui/material';
import { Loop, Timeline as TimelineIcon } from '@mui/icons-material';
import AgentStepCard from './AgentStepCard';
import { useIncidentStore } from '../../store/incidentStore';

// Groups streamed events by iteration and renders them as a vertical timeline.
export default function LoopTimeline({ events = [], streamConnected }) {
  const scrollRef = useRef(null);
  const { agentBusy, activity } = useIncidentStore();

  // Group events by iteration, preserving arrival order within a group.
  const groups = useMemo(() => {
    const map = new Map();
    events.forEach((evt, idx) => {
      const iter = typeof evt.iteration === 'number' ? evt.iteration : 0;
      if (!map.has(iter)) map.set(iter, []);
      map.get(iter).push({ evt, idx });
    });
    return Array.from(map.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([iteration, items]) => ({ iteration, items }));
  }, [events]);

  // Auto scroll to newest event as the loop streams in. `agentBusy` is included so
  // the pending-work row at the bottom scrolls into view too - otherwise the one
  // element that proves the system is alive is the one hidden below the fold.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [events.length, agentBusy]);

  return (
    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardContent sx={{ pb: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <TimelineIcon color="primary" />
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            Loop Timeline
          </Typography>
          <Chip
            size="small"
            variant="outlined"
            color={streamConnected ? 'success' : 'default'}
            label={streamConnected ? 'streaming' : 'idle'}
          />
        </Box>
        <Typography variant="caption" color="text.secondary">
          Triage, Hypothesize, Diagnose, Remediate, Verify, repeat
        </Typography>
      </CardContent>
      <Divider />

      <Box
        ref={scrollRef}
        sx={{
          flexGrow: 1,
          overflowY: 'auto',
          p: 2,
          minHeight: 320,
        }}
      >
        {events.length === 0 ? (
          <EmptyTimeline />
        ) : (
          <Stack spacing={3}>
            {groups.map((group) => (
              <Box key={group.iteration}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
                  <Box
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 0.5,
                      px: 1,
                      py: 0.25,
                      borderRadius: 999,
                      bgcolor: 'action.selected',
                      // info.main rather than info.light: the light palette's
                      // light shade is a pale sky blue that vanishes on white.
                      color: 'info.main',
                    }}
                  >
                    <Loop sx={{ fontSize: 16 }} />
                    <Typography variant="caption" sx={{ fontWeight: 700 }}>
                      Iteration {group.iteration}
                    </Typography>
                  </Box>
                  <Divider sx={{ flexGrow: 1 }} />
                </Box>

                {/* A continuous rail behind the cards rather than a dashed border
                    on the stack: it reads as one thread of work, and the gaps
                    between cards stop looking like breaks in the run. */}
                <Box sx={{ position: 'relative', pl: 2.5 }}>
                  <Box
                    sx={{
                      position: 'absolute',
                      left: 7,
                      top: 8,
                      bottom: 8,
                      width: 3,
                      borderRadius: 2,
                      bgcolor: 'divider',
                    }}
                  />
                  <Stack spacing={1.5}>
                    {group.items.map(({ evt, idx }) => (
                      <AgentStepCard key={idx} event={evt} />
                    ))}
                  </Stack>
                </Box>
              </Box>
            ))}

            {agentBusy && (
              <Box
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1.5,
                  p: 1.5,
                  borderRadius: 2,
                  border: '1px dashed',
                  borderColor: 'info.main',
                  bgcolor: 'rgba(56,189,248,0.06)',
                }}
              >
                <CircularProgress size={18} />
                <Typography variant="body2" color="text.secondary">
                  {activity || 'Waiting on the model...'}
                </Typography>
              </Box>
            )}
          </Stack>
        )}
      </Box>
    </Card>
  );
}

function EmptyTimeline() {
  return (
    <Box
      sx={{
        height: '100%',
        minHeight: 280,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        color: 'text.secondary',
        gap: 1,
      }}
    >
      <TimelineIcon sx={{ fontSize: 48, opacity: 0.4 }} />
      <Typography variant="body1">No active incident</Typography>
      <Typography variant="body2" sx={{ maxWidth: 360 }}>
        Pick a fault scenario on the left and trigger an incident. The agent loop will stream its
        steps here in real time.
      </Typography>
    </Box>
  );
}
