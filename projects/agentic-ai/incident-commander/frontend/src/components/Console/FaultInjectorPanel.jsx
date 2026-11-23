import { useEffect, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Divider,
  FormControl,
  FormControlLabel,
  InputLabel,
  ListSubheader,
  MenuItem,
  Select,
  Stack,
  Switch,
  Typography,
} from '@mui/material';
import { Bolt, ScienceOutlined, Refresh } from '@mui/icons-material';
import { useIncidentStore, TERMINAL_STATUSES } from '../../store/incidentStore';

function severityColor(severity) {
  switch ((severity || '').toLowerCase()) {
    case 'sev1':
    case 'critical':
    case 'high':
      return 'error';
    case 'sev2':
    case 'medium':
      return 'warning';
    default:
      return 'info';
  }
}

// Scenarios whose correct answer is to NOT act. Called out in the picker because
// they are the most interesting ones to run and the easiest to mistake for a bug
// when the agent "does nothing".
const NEGATIVE_SCENARIOS = new Set(['unfixable_ca_expiry', 'false_alarm']);
const ADVERSARIAL_SCENARIOS = new Set(['poisoned_telemetry']);

export default function FaultInjectorPanel() {
  const {
    scenarios,
    scenariosLoading,
    scenariosError,
    loadScenarios,
    triggerIncident,
    triggering,
    incidentId,
    status,
  } = useIncidentStore();

  const [pickedId, setSelectedId] = useState('');
  const [autoApprove, setAutoApprove] = useState(false);

  useEffect(() => {
    loadScenarios();
  }, [loadScenarios]);

  // Preselect the scenario worth demoing first, so the panel is never a dead end.
  // Derived during render rather than copied into state by an effect.
  const selectedId =
    pickedId || scenarios.find((s) => s.id === 'bad_deploy')?.id || scenarios[0]?.id || '';

  // Uses the shared terminal list rather than a local one: an earlier hand-written
  // copy omitted "cancelled", so cancelling an incident left the trigger button
  // permanently disabled with no way back except a page reload.
  const activeRunning = Boolean(incidentId && status && !TERMINAL_STATUSES.includes(status));

  const selected = scenarios.find((s) => s.id === selectedId) || null;

  const handleTrigger = async () => {
    if (!selectedId) return;
    await triggerIncident({
      scenarioId: selectedId,
      service: selected?.default_service,
      autoApprove,
    });
  };

  return (
    <Card sx={{ display: 'flex', flexDirection: 'column' }}>
      <CardContent sx={{ pb: 1.5 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <ScienceOutlined color="warning" />
          <Typography variant="h6" sx={{ flexGrow: 1, minWidth: 0 }}>
            Fault Injector
          </Typography>
          <Button
            size="small"
            onClick={loadScenarios}
            disabled={scenariosLoading}
            sx={{ minWidth: 0 }}
          >
            <Refresh fontSize="small" />
          </Button>
        </Box>
        <Typography variant="caption" color="text.secondary">
          Inject a simulated production fault, then watch the agent work it.
        </Typography>
      </CardContent>
      <Divider />

      {/* A dropdown, not a scrolling list of ten cards, and it grows with its
          content rather than scrolling inside a fixed-height box. Both were the
          same bug twice: a narrow column plus a viewport-locked height collapsed
          this region to about one line, so the scenario name was clipped mid-word
          and its description - the whole point of the picker - sat behind a
          60px scrollbar. */}
      <Box sx={{ p: 2 }}>
        {scenariosError ? (
          <Alert severity="error" variant="outlined">
            {scenariosError}
          </Alert>
        ) : (
          <Stack spacing={2}>
            <FormControl fullWidth size="small" disabled={scenariosLoading || activeRunning}>
              <InputLabel id="scenario-label">Fault scenario</InputLabel>
              <Select
                labelId="scenario-label"
                label="Fault scenario"
                value={selectedId}
                onChange={(e) => setSelectedId(e.target.value)}
              >
                <ListSubheader>Faults the agent should fix</ListSubheader>
                {scenarios
                  .filter((s) => !NEGATIVE_SCENARIOS.has(s.id) && !ADVERSARIAL_SCENARIOS.has(s.id))
                  .map((s) => (
                    <MenuItem key={s.id} value={s.id}>
                      {s.name}
                    </MenuItem>
                  ))}
                <ListSubheader>Adversarial</ListSubheader>
                {scenarios
                  .filter((s) => ADVERSARIAL_SCENARIOS.has(s.id))
                  .map((s) => (
                    <MenuItem key={s.id} value={s.id}>
                      {s.name}
                    </MenuItem>
                  ))}
                <ListSubheader>Correct answer is to NOT act</ListSubheader>
                {scenarios
                  .filter((s) => NEGATIVE_SCENARIOS.has(s.id))
                  .map((s) => (
                    <MenuItem key={s.id} value={s.id}>
                      {s.name}
                    </MenuItem>
                  ))}
              </Select>
            </FormControl>

            {selected && (
              <Box>
                <Stack direction="row" spacing={0.75} sx={{ mb: 1, flexWrap: 'wrap', gap: 0.75 }}>
                  {selected.severity && (
                    <Chip
                      size="small"
                      variant="outlined"
                      color={severityColor(selected.severity)}
                      label={selected.severity}
                    />
                  )}
                  {selected.default_service && (
                    <Chip size="small" variant="outlined" label={selected.default_service} />
                  )}
                  {NEGATIVE_SCENARIOS.has(selected.id) && (
                    <Chip size="small" color="info" label="doing nothing is correct" />
                  )}
                  {ADVERSARIAL_SCENARIOS.has(selected.id) && (
                    <Chip size="small" color="warning" label="contains injected text" />
                  )}
                </Stack>
                <Typography variant="body2" color="text.secondary">
                  {selected.description}
                </Typography>
                {NEGATIVE_SCENARIOS.has(selected.id) && (
                  <Typography
                    variant="caption"
                    sx={{ color: 'info.light', display: 'block', mt: 1 }}
                  >
                    Expect an escalation, not a fix. An agent that invents a remedy here has failed
                    the scenario even though it looks busy.
                  </Typography>
                )}
              </Box>
            )}
          </Stack>
        )}
      </Box>

      <Divider />
      <CardContent>
        <FormControlLabel
          sx={{ alignItems: 'flex-start', ml: 0 }}
          control={
            <Switch
              checked={autoApprove}
              onChange={(e) => setAutoApprove(e.target.checked)}
              color="warning"
              size="small"
              sx={{ mr: 1 }}
            />
          }
          label={
            <Box>
              <Typography variant="body2">Auto-approve low/medium risk</Typography>
              <Typography variant="caption" color="text.secondary">
                High-risk actions still stop for you, always.
              </Typography>
            </Box>
          }
        />
        <Button
          fullWidth
          variant="contained"
          size="large"
          startIcon={
            triggering || activeRunning ? <CircularProgress size={18} color="inherit" /> : <Bolt />
          }
          disabled={!selectedId || triggering || activeRunning}
          onClick={handleTrigger}
          sx={{ mt: 1.5 }}
        >
          {triggering
            ? 'Starting...'
            : activeRunning
              ? 'Running - watch above'
              : 'Trigger Incident'}
        </Button>
        {/* A live run is mostly model latency, and the button going quiet reads as
            failure. Saying the number up front stops the reflex second press. */}
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ display: 'block', mt: 1, textAlign: 'center' }}
        >
          {activeRunning
            ? 'Progress streams into the status bar above.'
            : 'About 30-90 seconds, mostly model latency. Roughly $0.01-0.03.'}
        </Typography>
      </CardContent>
    </Card>
  );
}
