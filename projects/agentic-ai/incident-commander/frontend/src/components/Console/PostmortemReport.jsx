import {
  Box,
  Card,
  CardContent,
  Chip,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Stack,
  Typography,
} from '@mui/material';
import {
  Description,
  ReportProblem,
  Build,
  FactCheck,
  Lightbulb,
  AccessTime,
} from '@mui/icons-material';
import { formatTime } from './agentMeta';

function Section({ icon, title, children }) {
  return (
    <Box sx={{ mb: 2.5 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
        {icon}
        <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
          {title}
        </Typography>
      </Box>
      {children}
    </Box>
  );
}

// Renders the postmortem object emitted by the agent.
export default function PostmortemReport({ postmortem }) {
  if (!postmortem) return null;

  const {
    summary,
    root_cause,
    timeline = [],
    actions_taken = [],
    evidence = [],
    recommendations = [],
  } = postmortem;

  return (
    <Card sx={{ borderTop: '3px solid', borderColor: 'info.main' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
          <Description color="info" />
          <Typography variant="h6">Postmortem</Typography>
        </Box>

        {summary && (
          <Section icon={<Description fontSize="small" color="info" />} title="Summary">
            <Typography variant="body2" color="text.secondary">
              {summary}
            </Typography>
          </Section>
        )}

        {root_cause && (
          <Section icon={<ReportProblem fontSize="small" color="error" />} title="Root cause">
            <Box
              sx={{
                p: 1.5,
                borderRadius: 1.5,
                bgcolor: 'rgba(248, 113, 113, 0.08)',
                border: '1px solid',
                borderColor: 'rgba(248, 113, 113, 0.24)',
              }}
            >
              <Typography variant="body2">{root_cause}</Typography>
            </Box>
          </Section>
        )}

        {timeline.length > 0 && (
          <Section icon={<AccessTime fontSize="small" color="primary" />} title="Timeline">
            <List dense disablePadding>
              {timeline.map((t, i) => (
                <ListItem key={i} sx={{ alignItems: 'flex-start', px: 0 }}>
                  <Chip
                    size="small"
                    variant="outlined"
                    label={formatTime(t.ts)}
                    sx={{ mr: 1.5, mt: 0.25, fontFamily: 'ui-monospace, monospace' }}
                  />
                  {/* The label alone ("Diagnostician: diagnostic") describes the
                      shape of the run with none of its content. `detail` carries
                      what was actually decided at that step - the cause and its
                      confidence, the objection, whether the prediction held. */}
                  <ListItemText
                    primary={t.event}
                    secondary={t.detail || null}
                    slotProps={{
                      primary: { variant: 'body2', fontWeight: 600 },
                      secondary: { variant: 'body2', color: 'text.secondary' },
                    }}
                  />
                </ListItem>
              ))}
            </List>
          </Section>
        )}

        {actions_taken.length > 0 && (
          <Section icon={<Build fontSize="small" color="warning" />} title="Actions taken">
            <Stack spacing={0.75}>
              {actions_taken.map((a, i) => (
                <Box
                  key={i}
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1,
                    p: 1,
                    borderRadius: 1,
                    bgcolor: 'rgba(148, 163, 184, 0.06)',
                  }}
                >
                  <Build sx={{ fontSize: 16, color: 'warning.main' }} />
                  <Typography variant="body2">{renderAction(a)}</Typography>
                </Box>
              ))}
            </Stack>
          </Section>
        )}

        {evidence.length > 0 && (
          <Section icon={<FactCheck fontSize="small" color="success" />} title="Evidence">
            <List dense disablePadding>
              {evidence.map((e, i) => (
                <ListItem key={i} sx={{ alignItems: 'flex-start', px: 0 }}>
                  <ListItemIcon sx={{ minWidth: 32, mt: 0.5 }}>
                    <FactCheck sx={{ fontSize: 18, color: 'success.main' }} />
                  </ListItemIcon>
                  <ListItemText
                    primary={e.source}
                    secondary={e.detail}
                    slotProps={{
                      primary: { variant: 'body2', fontWeight: 600 },
                      secondary: { variant: 'body2' },
                    }}
                  />
                </ListItem>
              ))}
            </List>
          </Section>
        )}

        {recommendations.length > 0 && (
          <>
            <Divider sx={{ my: 1.5 }} />
            <Section
              icon={<Lightbulb fontSize="small" color="secondary" />}
              title="Recommendations"
            >
              <List dense disablePadding>
                {recommendations.map((r, i) => (
                  <ListItem key={i} sx={{ alignItems: 'flex-start', px: 0 }}>
                    <ListItemIcon sx={{ minWidth: 32, mt: 0.5 }}>
                      <Lightbulb sx={{ fontSize: 18, color: 'secondary.main' }} />
                    </ListItemIcon>
                    <ListItemText
                      primary={typeof r === 'string' ? r : r.text || JSON.stringify(r)}
                      slotProps={{ primary: { variant: 'body2' } }}
                    />
                  </ListItem>
                ))}
              </List>
            </Section>
          </>
        )}
      </CardContent>
    </Card>
  );
}

function renderAction(a) {
  if (typeof a === 'string') return a;
  if (!a || typeof a !== 'object') return String(a);
  const parts = [];
  if (a.action_type) parts.push(a.action_type);
  if (a.target) parts.push(`on ${a.target}`);
  if (a.result || a.status) parts.push(`(${a.result || a.status})`);
  const who = decidedBy(a);
  if (who) parts.push(`- ${who}`);
  return parts.join(' ') || JSON.stringify(a);
}

// The audit line: who let this action run, or who stopped it.
export function decidedBy(a) {
  const by = a?.decided_by;
  if (!by) return '';
  if (by === 'policy:auto_approve') return 'auto-approved by policy';
  if (by === 'nobody:timeout') return 'no decision (approval timed out)';
  const verb = a.status === 'rejected' ? 'rejected' : 'approved';
  return `${verb} by ${by}`;
}
