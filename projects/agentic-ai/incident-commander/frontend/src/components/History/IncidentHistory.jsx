import { useEffect, useState } from 'react';
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
import { incidentAPI } from '../../services/api';
import { statusColor, formatTime } from '../Console/agentMeta';
import PostmortemReport from '../Console/PostmortemReport';

export default function IncidentHistory() {
  const [incidents, setIncidents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [selected, setSelected] = useState(null); // full incident object
  const [detailLoading, setDetailLoading] = useState(false);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await incidentAPI.listIncidents();
      setIncidents(data.incidents || []);
    } catch (err) {
      setError(err?.message || 'Failed to load incidents');
    } finally {
      setLoading(false);
    }
  };

  // First load. Written out rather than calling load(), whose synchronous
  // setLoading(true) inside an effect would trigger an extra render.
  useEffect(() => {
    let active = true;
    incidentAPI
      .listIncidents()
      .then((data) => active && setIncidents(data.incidents || []))
      .catch((err) => active && setError(err?.message || 'Failed to load incidents'))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, []);

  const openIncident = async (incidentId) => {
    setDetailLoading(true);
    setSelected({ incident_id: incidentId, postmortem: null });
    try {
      const data = await incidentAPI.getIncident(incidentId);
      setSelected(data);
    } catch (err) {
      setError(err?.message || 'Failed to load incident detail');
      setSelected(null);
    } finally {
      setDetailLoading(false);
    }
  };

  return (
    <Box>
      <Card>
        <CardContent sx={{ pb: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <HistoryIcon color="primary" />
            <Typography variant="h6" sx={{ flexGrow: 1 }}>
              Past Incidents
            </Typography>
            <Button size="small" startIcon={<Refresh />} onClick={load} disabled={loading}>
              Refresh
            </Button>
          </Box>
          <Typography variant="caption" color="text.secondary">
            Click a row to view its postmortem.
          </Typography>
        </CardContent>

        {error && (
          <Box sx={{ px: 2, pb: 2 }}>
            <Alert severity="error" variant="outlined">
              {error}
            </Alert>
          </Box>
        )}

        {loading && incidents.length === 0 ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 6 }}>
            <CircularProgress />
          </Box>
        ) : incidents.length === 0 ? (
          <Box sx={{ textAlign: 'center', py: 6, color: 'text.secondary' }}>
            <HistoryIcon sx={{ fontSize: 48, opacity: 0.4, mb: 1 }} />
            <Typography variant="body1">No incidents yet</Typography>
            <Typography variant="body2">Trigger one from the Console to see it here.</Typography>
          </Box>
        ) : (
          <TableContainer component={Paper} sx={{ boxShadow: 'none', bgcolor: 'transparent' }}>
            <Table size="small">
              {/* A tinted, bold header row. Without it the header is just the
                  first data row in a slightly different weight, and a long table
                  loses its anchor as soon as you scroll. */}
              <TableHead>
                <TableRow
                  sx={{
                    bgcolor: 'action.hover',
                    '& th': { fontWeight: 700, whiteSpace: 'nowrap' },
                  }}
                >
                  <TableCell>Incident</TableCell>
                  <TableCell>Scenario</TableCell>
                  <TableCell>Service</TableCell>
                  <TableCell>Severity</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell align="right">Iterations</TableCell>
                  <TableCell>Started</TableCell>
                  <TableCell align="right" />
                </TableRow>
              </TableHead>
              <TableBody>
                {incidents.map((inc) => (
                  <TableRow
                    key={inc.incident_id}
                    hover
                    sx={{ cursor: 'pointer' }}
                    onClick={() => openIncident(inc.incident_id)}
                  >
                    <TableCell sx={{ fontFamily: 'ui-monospace, monospace', fontSize: 12 }}>
                      {inc.incident_id}
                    </TableCell>
                    <TableCell>{inc.scenario_id}</TableCell>
                    <TableCell>{inc.service || '-'}</TableCell>
                    <TableCell>
                      {inc.severity ? (
                        <Chip size="small" variant="outlined" label={inc.severity} />
                      ) : (
                        '-'
                      )}
                    </TableCell>
                    <TableCell>
                      <Chip
                        size="small"
                        color={statusColor(inc.status)}
                        label={(inc.status || '').replace(/_/g, ' ')}
                        variant={inc.resolved ? 'filled' : 'outlined'}
                      />
                    </TableCell>
                    <TableCell align="right">{inc.iterations ?? 0}</TableCell>
                    <TableCell>{formatTime(inc.started_at)}</TableCell>
                    <TableCell align="right">
                      <IconButton
                        size="small"
                        onClick={(e) => {
                          e.stopPropagation();
                          openIncident(inc.incident_id);
                        }}
                      >
                        <Visibility fontSize="small" />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </Card>

      <Dialog
        open={Boolean(selected)}
        onClose={() => setSelected(null)}
        maxWidth="md"
        fullWidth
        scroll="paper"
      >
        <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box sx={{ flexGrow: 1 }}>
            Incident{' '}
            <Box component="span" sx={{ fontFamily: 'ui-monospace, monospace', fontSize: 14 }}>
              {selected?.incident_id}
            </Box>
          </Box>
          {selected?.status && (
            <Chip
              size="small"
              color={statusColor(selected.status)}
              label={(selected.status || '').replace(/_/g, ' ')}
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
          ) : selected?.postmortem ? (
            <PostmortemReport postmortem={selected.postmortem} />
          ) : (
            <Box sx={{ textAlign: 'center', py: 6, color: 'text.secondary' }}>
              <Typography variant="body1">No postmortem available</Typography>
              <Typography variant="body2">
                This incident did not produce a postmortem (it may still be running or was not
                resolved).
              </Typography>
            </Box>
          )}
        </DialogContent>
      </Dialog>
    </Box>
  );
}
