import React, { useEffect, useMemo, useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Dialog,
  DialogContent,
  DialogTitle,
  IconButton,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
  Alert,
} from '@mui/material';
import { Refresh, Close, History as HistoryIcon, Visibility } from '@mui/icons-material';
import { runAPI } from '../../services/api';
import { projectRun } from '../../store/migrationStore';
import { jobStatusColor, modeColor, formatUsd, formatTime } from '../Console/format';
import FleetBoard from '../Console/FleetBoard';
import JobSummaryBanner from '../Console/JobSummaryBanner';

// The runs list identifies a run by run_id; tolerate id as a fallback.
function runIdOf(run) {
  return run?.run_id ?? run?.id ?? null;
}

export default function MigrationHistory() {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [selected, setSelected] = useState(null); // { runId, data }
  const [detailLoading, setDetailLoading] = useState(false);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await runAPI.listRuns();
      setRuns(data.runs || []);
    } catch (err) {
      setError(err?.message || 'Failed to load runs');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const openRun = async (runId) => {
    if (!runId) return;
    setDetailLoading(true);
    setSelected({ runId, data: null });
    try {
      const data = await runAPI.getRun(runId);
      setSelected({ runId, data });
    } catch (err) {
      setError(err?.message || 'Failed to load run detail');
      setSelected(null);
    } finally {
      setDetailLoading(false);
    }
  };

  // Derive the read-only projection for the selected run (fleet board + summary).
  const projection = useMemo(
    () => (selected?.data ? projectRun(selected.data) : null),
    [selected]
  );
  const repoList = projection
    ? projection.reposOrder.map((id) => projection.repos[id]).filter(Boolean)
    : [];

  return (
    <Box>
      <Card>
        <CardContent sx={{ pb: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <HistoryIcon color="primary" />
            <Typography variant="h6" sx={{ flexGrow: 1 }}>
              Past migration runs
            </Typography>
            <Button size="small" startIcon={<Refresh />} onClick={load} disabled={loading}>
              Refresh
            </Button>
          </Box>
          <Typography variant="caption" color="text.secondary">
            Click a row to replay its fleet board and summary.
          </Typography>
        </CardContent>

        {error && (
          <Box sx={{ px: 2, pb: 2 }}>
            <Alert severity="error" variant="outlined">
              {error}
            </Alert>
          </Box>
        )}

        {loading && runs.length === 0 ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 6 }}>
            <CircularProgress />
          </Box>
        ) : runs.length === 0 ? (
          <Box sx={{ textAlign: 'center', py: 6, color: 'text.secondary' }}>
            <HistoryIcon sx={{ fontSize: 48, opacity: 0.4, mb: 1 }} />
            <Typography variant="body1">No runs yet</Typography>
            <Typography variant="body2">Launch one from the Console to see it here.</Typography>
          </Box>
        ) : (
          <TableContainer component={Paper} sx={{ boxShadow: 'none', bgcolor: 'transparent' }}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Title</TableCell>
                  <TableCell>Rule</TableCell>
                  <TableCell>Mode</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell align="right">Merged</TableCell>
                  <TableCell align="right">Cost</TableCell>
                  <TableCell>Started</TableCell>
                  <TableCell align="right" />
                </TableRow>
              </TableHead>
              <TableBody>
                {runs.map((run, i) => {
                  const id = runIdOf(run);
                  return (
                    <TableRow
                      key={id || i}
                      hover
                      sx={{ cursor: id ? 'pointer' : 'default' }}
                      onClick={() => openRun(id)}
                    >
                      <TableCell>{run.title || '-'}</TableCell>
                      <TableCell sx={{ fontFamily: 'ui-monospace, monospace', fontSize: 12 }}>
                        {run.rule_id || '-'}
                      </TableCell>
                      <TableCell>
                        {run.mode ? (
                          <Chip
                            size="small"
                            variant="outlined"
                            color={modeColor(run.mode)}
                            label={run.mode}
                          />
                        ) : (
                          '-'
                        )}
                      </TableCell>
                      <TableCell>
                        <Chip
                          size="small"
                          color={jobStatusColor(run.status)}
                          label={(run.status || '').replace(/_/g, ' ')}
                          variant={['done', 'failed'].includes(run.status) ? 'filled' : 'outlined'}
                        />
                      </TableCell>
                      <TableCell align="right">
                        {run.repos_merged ?? 0} / {run.repos_total ?? 0}
                      </TableCell>
                      <TableCell align="right">{formatUsd(run.total_cost_usd)}</TableCell>
                      <TableCell>{formatTime(run.started_at)}</TableCell>
                      <TableCell align="right">
                        <IconButton
                          size="small"
                          disabled={!id}
                          onClick={(e) => {
                            e.stopPropagation();
                            openRun(id);
                          }}
                        >
                          <Visibility fontSize="small" />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </Card>

      <Dialog
        open={Boolean(selected)}
        onClose={() => setSelected(null)}
        maxWidth="lg"
        fullWidth
        scroll="paper"
      >
        <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box sx={{ flexGrow: 1 }}>
            {projection?.title || 'Migration run'}{' '}
            <Box component="span" sx={{ fontFamily: 'ui-monospace, monospace', fontSize: 13 }}>
              {selected?.runId}
            </Box>
          </Box>
          {projection?.jobStatus && (
            <Chip
              size="small"
              color={jobStatusColor(projection.jobStatus)}
              label={(projection.jobStatus || '').replace(/_/g, ' ')}
            />
          )}
          <IconButton onClick={() => setSelected(null)} size="small">
            <Close fontSize="small" />
          </IconButton>
        </DialogTitle>
        <DialogContent dividers>
          {detailLoading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 6 }}>
              <CircularProgress />
            </Box>
          ) : projection ? (
            <Stack spacing={2}>
              <JobSummaryBanner summary={projection.summary} />
              <Box>
                <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
                  Fleet board (read-only replay)
                </Typography>
                <FleetBoard repos={repoList} readOnly emptyHint="This run recorded no repositories." />
              </Box>
            </Stack>
          ) : (
            <Box sx={{ textAlign: 'center', py: 6, color: 'text.secondary' }}>
              <Typography variant="body1">No detail available</Typography>
            </Box>
          )}
        </DialogContent>
      </Dialog>
    </Box>
  );
}
