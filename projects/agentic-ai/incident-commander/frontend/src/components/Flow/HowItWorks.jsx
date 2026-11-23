import { useState } from 'react';
import { Box, Button, Card, CardContent, Grid, Stack, Tab, Tabs, Typography } from '@mui/material';
import { Bolt, Shield, VisibilityOff, Gavel } from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import { AGENT_META } from '../Console/agentMeta';
import { toneFor, tintFor } from '../../theme';
import FlowDiagram from './FlowDiagram';
import LoopAnimation from './LoopAnimation';

// Shown when there is nothing to trace yet: what this system does, end to end.
//
// It lives here rather than behind a docs link because the question it answers -
// "what am I looking at?" - arrives exactly when the trace view is empty. It used
// to be four cards stacked down the right-hand side; the same content is now one
// tabbed panel, so the diagram stays on screen while you read about it instead of
// scrolling away above the prose.
export default function HowItWorks({ onGoToConsole }) {
  const mode = useTheme().palette.mode;
  const [tab, setTab] = useState(0);

  return (
    <Grid container spacing={2} sx={{ alignItems: 'flex-start' }}>
      <Grid size={{ xs: 12, md: 5 }}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              How one incident flows
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              An alert fires on a simulated stack. Nothing is scripted: the agents never see the
              correct fix and have to infer it from telemetry. Hover any step.
            </Typography>
            <LoopAnimation height={300} />
            <Box sx={{ mt: 1.5 }}>
              <FlowDiagram counts={{}} />
            </Box>
          </CardContent>
        </Card>
      </Grid>

      <Grid size={{ xs: 12, md: 7 }}>
        <Card>
          <CardContent sx={{ pb: 0 }}>
            <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
              The model decides what to do. Code decides what is allowed.
            </Typography>
            <Typography variant="body2" color="text.secondary">
              The LLM routes, diagnoses, proposes, critiques and verifies. It never sets a ceiling,
              grants a permission, classifies its own risk, or waives an approval. Telemetry is
              attacker-influenced text that reaches every prompt, so anything a model can influence
              must not be able to widen the model&apos;s own permissions.
            </Typography>
          </CardContent>

          <Tabs
            value={tab}
            onChange={(_, v) => setTab(v)}
            variant="scrollable"
            scrollButtons="auto"
            sx={{ px: 2, mt: 1, borderBottom: '1px solid', borderColor: 'divider' }}
          >
            <Tab label="Who is in the room" />
            <Tab label="What to watch for" />
          </Tabs>

          <CardContent>
            {tab === 0 && (
              <Grid container spacing={1.5}>
                {Object.entries(AGENT_META).map(([name, meta]) => (
                  <Grid size={{ xs: 12, sm: 6 }} key={name}>
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
              <Stack spacing={1.5}>
                <Point
                  icon={<Gavel fontSize="small" />}
                  title="It stops and waits for you"
                  body="High-risk actions always pause for a human, even with auto-approve on. The loop is genuinely blocked - not polling, not pretending."
                />
                <Point
                  icon={<Shield fontSize="small" />}
                  title="Injected telemetry is removed, not flagged"
                  body="In Poisoned Telemetry, attacker text demanding a destructive failover is stripped before any agent sees it, and the attempt is recorded."
                />
                <Point
                  icon={<VisibilityOff fontSize="small" />}
                  title="Sometimes doing nothing is correct"
                  body="Unfixable CA Expiry and False Alarm are scored on the agent NOT acting. An agent measured only on faults it can fix will invent a fix for one it cannot."
                />
                <Point
                  icon={<Bolt fontSize="small" />}
                  title="Every prediction is falsifiable"
                  body="Each fix predicts a specific signal. The Verifier checks it against fresh telemetry, so a lucky recovery is flagged rather than counted as a clean win."
                />
              </Stack>
            )}

            <Box sx={{ mt: 2.5 }}>
              <Button variant="contained" startIcon={<Bolt />} onClick={onGoToConsole}>
                Run one - Bad Deploy, ~30s
              </Button>
            </Box>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
}

function Point({ icon, title, body }) {
  return (
    <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'flex-start' }}>
      <Box sx={{ color: 'info.main', mt: 0.3 }}>{icon}</Box>
      <Box>
        <Typography variant="subtitle2">{title}</Typography>
        <Typography variant="body2" color="text.secondary">
          {body}
        </Typography>
      </Box>
    </Box>
  );
}
