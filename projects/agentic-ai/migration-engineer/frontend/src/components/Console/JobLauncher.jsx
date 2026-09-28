import React, { useEffect, useState } from 'react';
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
  MenuItem,
  Select,
  Stack,
  Switch,
  Typography,
} from '@mui/material';
import { PlayArrow, Rule, FolderOutlined } from '@mui/icons-material';
import { useMigrationStore } from '../../store/migrationStore';

const RUNNING_STATUSES = ['planning', 'running'];

// Left-hand launcher: pick a migration job, optionally auto-approve, then run it.
export default function JobLauncher() {
  const {
    jobs,
    jobsLoading,
    jobsError,
    loadJobs,
    triggerRun,
    triggering,
    jobStatus,
    streamConnected,
  } = useMigrationStore();

  const [selectedId, setSelectedId] = useState('');
  const [autoApprove, setAutoApprove] = useState(false);

  useEffect(() => {
    loadJobs();
  }, [loadJobs]);

  const selected = jobs.find((j) => j.id === selectedId) || null;
  const running = RUNNING_STATUSES.includes(jobStatus);

  const handleRun = async () => {
    if (!selectedId) return;
    await triggerRun({ jobId: selectedId, autoApprove });
  };

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
          <PlayArrow color="primary" />
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            Launch a migration
          </Typography>
          <Chip
            size="small"
            variant="outlined"
            color={streamConnected ? 'success' : 'default'}
            label={streamConnected ? 'streaming' : running ? 'starting' : 'idle'}
          />
        </Box>
        <Typography variant="caption" color="text.secondary">
          Select a codemod job, then let the agent fleet migrate every repository in parallel.
        </Typography>

        <Divider sx={{ my: 2 }} />

        {jobsError && (
          <Alert severity="error" variant="outlined" sx={{ mb: 2 }}>
            {jobsError}
          </Alert>
        )}

        <Stack spacing={2}>
          <FormControl fullWidth size="small" disabled={jobsLoading || running}>
            <InputLabel id="job-select-label">Migration job</InputLabel>
            <Select
              labelId="job-select-label"
              label="Migration job"
              value={selectedId}
              onChange={(e) => setSelectedId(e.target.value)}
              renderValue={(value) => {
                const job = jobs.find((j) => j.id === value);
                return job ? job.title : '';
              }}
            >
              {jobs.length === 0 && (
                <MenuItem value="" disabled>
                  {jobsLoading ? 'Loading jobs...' : 'No jobs available'}
                </MenuItem>
              )}
              {jobs.map((job) => (
                <MenuItem key={job.id} value={job.id}>
                  <Box>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="body2" fontWeight={600}>
                        {job.title}
                      </Typography>
                      <Chip size="small" variant="outlined" label={`${job.repo_count} repos`} />
                    </Box>
                    <Typography variant="caption" color="text.secondary">
                      {job.rule_name}
                      {job.description ? ` - ${job.description}` : ''}
                    </Typography>
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {selected && (
            <Box
              sx={{
                p: 1.5,
                borderRadius: 1.5,
                bgcolor: 'rgba(148, 163, 184, 0.06)',
                border: '1px solid',
                borderColor: 'divider',
              }}
            >
              <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5 }}>
                <Rule sx={{ fontSize: 16, color: 'primary.main' }} />
                <Typography variant="body2" fontWeight={600}>
                  {selected.rule_name}
                </Typography>
                <Chip
                  size="small"
                  variant="outlined"
                  label={selected.rule_id}
                  sx={{ fontFamily: 'ui-monospace, monospace' }}
                />
              </Stack>
              {selected.description && (
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  {selected.description}
                </Typography>
              )}
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, flexWrap: 'wrap' }}>
                <FolderOutlined sx={{ fontSize: 16, color: 'text.secondary' }} />
                {(selected.repo_ids || []).map((rid) => (
                  <Chip
                    key={rid}
                    size="small"
                    variant="outlined"
                    label={rid}
                    sx={{ fontFamily: 'ui-monospace, monospace' }}
                  />
                ))}
              </Box>
            </Box>
          )}

          <FormControlLabel
            control={
              <Switch
                checked={autoApprove}
                onChange={(e) => setAutoApprove(e.target.checked)}
                color="warning"
                disabled={running}
              />
            }
            label={
              <Box>
                <Typography variant="body2">Auto-approve PRs</Typography>
                <Typography variant="caption" color="text.secondary">
                  Skip the human gate and open PRs automatically
                </Typography>
              </Box>
            }
          />
          {autoApprove && (
            <Chip size="small" color="warning" label="Approval gate disabled" sx={{ alignSelf: 'flex-start' }} />
          )}

          <Button
            fullWidth
            variant="contained"
            size="large"
            startIcon={triggering ? <CircularProgress size={18} color="inherit" /> : <PlayArrow />}
            disabled={!selectedId || triggering || running}
            onClick={handleRun}
          >
            {running ? 'Migration in progress' : 'Run migration'}
          </Button>
        </Stack>
      </CardContent>
    </Card>
  );
}
