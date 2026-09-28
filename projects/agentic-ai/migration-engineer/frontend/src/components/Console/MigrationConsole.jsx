import React, { useEffect, useMemo } from 'react';
import { Alert, Box, Chip, Collapse, Divider, Grid, Stack, Typography } from '@mui/material';
import { Token, Paid } from '@mui/icons-material';
import { useMigrationStore, TERMINAL_JOB_STATUSES } from '../../store/migrationStore';
import JobLauncher from './JobLauncher';
import ApprovalGate from './ApprovalGate';
import JobSummaryBanner from './JobSummaryBanner';
import FleetBoard from './FleetBoard';
import { jobStatusColor, formatUsd } from './format';

function ActiveRunHeader() {
  const { title, jobStatus, mode, totalTokens, totalCostUsd, reposOrder } = useMigrationStore();
  if (!jobStatus) return null;

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 1.5,
        flexWrap: 'wrap',
        p: 1.5,
        borderRadius: 1.5,
        bgcolor: 'rgba(148, 163, 184, 0.06)',
        border: '1px solid',
        borderColor: 'divider',
      }}
    >
      <Typography variant="subtitle1" sx={{ flexGrow: 1, fontWeight: 700 }}>
        {title || 'Migration run'}
      </Typography>
      <Chip
        size="small"
        color={jobStatusColor(jobStatus)}
        label={jobStatus.replace(/_/g, ' ')}
        variant={['done', 'failed'].includes(jobStatus) ? 'filled' : 'outlined'}
      />
      {mode && <Chip size="small" variant="outlined" label={mode} />}
      <Divider orientation="vertical" flexItem />
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, color: 'text.secondary' }}>
        <Token sx={{ fontSize: 16 }} />
        <Typography variant="body2">{totalTokens.toLocaleString()} tokens</Typography>
      </Box>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, color: 'success.light' }}>
        <Paid sx={{ fontSize: 16 }} />
        <Typography variant="body2">{formatUsd(totalCostUsd)}</Typography>
      </Box>
      <Chip size="small" variant="outlined" label={`${reposOrder.length} repos`} />
    </Box>
  );
}

export default function MigrationConsole() {
  const { repos, reposOrder, pendingApprovals, summary, error, jobStatus } = useMigrationStore();

  // On mount: if a run is still in flight (the operator navigated away and back),
  // re-attach to its stream - the backend replays the full history on connect.
  // On unmount: close the stream.
  useEffect(() => {
    const state = useMigrationStore.getState();
    if (state.runId && state.jobStatus && !TERMINAL_JOB_STATUSES.includes(state.jobStatus)) {
      state.connectStream(state.runId);
    }
    return () => useMigrationStore.getState().disconnectStream();
  }, []);

  const repoList = useMemo(
    () => reposOrder.map((id) => repos[id]).filter(Boolean),
    [reposOrder, repos]
  );

  const pendingList = useMemo(
    () => Object.entries(pendingApprovals),
    [pendingApprovals]
  );

  return (
    <Box>
      <Collapse in={Boolean(error)}>
        <Alert severity="error" variant="outlined" sx={{ mb: 2 }}>
          {error}
        </Alert>
      </Collapse>

      <Grid container spacing={2} alignItems="flex-start">
        {/* Left: launcher */}
        <Grid item xs={12} md={4} lg={3.5}>
          <JobLauncher />
        </Grid>

        {/* Right: active run */}
        <Grid item xs={12} md={8} lg={8.5}>
          <Stack spacing={2}>
            {jobStatus && <ActiveRunHeader />}

            <Collapse in={Boolean(summary)}>
              {summary && <JobSummaryBanner summary={summary} />}
            </Collapse>

            {pendingList.length > 0 && (
              <Box>
                <Typography variant="subtitle2" color="warning.main" sx={{ mb: 1 }}>
                  {pendingList.length} {pendingList.length === 1 ? 'repository is' : 'repositories are'}{' '}
                  awaiting your approval
                </Typography>
                <Stack spacing={2}>
                  {pendingList.map(([repoId, payload]) => (
                    <ApprovalGate key={repoId} repoId={repoId} payload={payload} />
                  ))}
                </Stack>
              </Box>
            )}

            <Box>
              <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
                Fleet board
              </Typography>
              <FleetBoard repos={repoList} />
            </Box>
          </Stack>
        </Grid>
      </Grid>
    </Box>
  );
}
