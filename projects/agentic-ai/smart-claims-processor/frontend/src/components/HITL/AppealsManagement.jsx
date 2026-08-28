import React, { useEffect, useState } from 'react';
import {
  Box, Paper, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Chip, Button, CircularProgress, Grid, Tabs, Tab, Dialog, DialogTitle, DialogContent,
  DialogActions, RadioGroup, Radio, FormControlLabel, TextField, Alert,
} from '@mui/material';
import { Gavel, Refresh, Visibility } from '@mui/icons-material';
import { appealsAPI } from '../../services/api';

const STATUS_COLORS = { pending: 'warning', upheld: 'error', overturned: 'success', partial: 'info' };

export default function AppealsManagement() {
  const [appeals, setAppeals] = useState([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState(0);
  const [selected, setSelected] = useState(null);
  const [decision, setDecision] = useState('');
  const [reasoning, setReasoning] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const fetchAppeals = async () => {
    setLoading(true);
    try {
      const res = await appealsAPI.getAllAppeals(null, 100);
      setAppeals(res.data.appeals || res.data || []);
    } catch { setAppeals([]); }
    finally { setLoading(false); }
  };

  useEffect(() => { fetchAppeals(); }, []);

  const tabFilters = ['all', 'pending', 'upheld', 'overturned'];
  const filtered = appeals.filter((a) => tab === 0 || a.status === tabFilters[tab]);

  const stats = {
    total: appeals.length,
    pending: appeals.filter((a) => a.status === 'pending').length,
    upheld: appeals.filter((a) => a.status === 'upheld').length,
    overturned: appeals.filter((a) => a.status === 'overturned').length,
  };

  const handleReview = async () => {
    setSubmitting(true);
    try {
      await appealsAPI.reviewAppeal(selected.appeal_id || selected.id, { decision, reasoning });
      setSelected(null);
      setDecision('');
      setReasoning('');
      fetchAppeals();
    } catch { /* handle */ }
    finally { setSubmitting(false); }
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h5" fontWeight={700} color="#F1F5F9">
            <Gavel sx={{ mr: 1, verticalAlign: 'middle', color: '#9C27B0' }} />
            Appeals Review
          </Typography>
          <Typography variant="body2" color="#94A3B8">Review claimant appeals against claim decisions</Typography>
        </Box>
        <Button startIcon={<Refresh />} onClick={fetchAppeals} sx={{ color: '#94A3B8' }}>Refresh</Button>
      </Box>

      {/* Stats */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {[
          { label: 'Total Appeals', value: stats.total, color: '#3B82F6' },
          { label: 'Pending', value: stats.pending, color: '#F59E0B' },
          { label: 'Upheld', value: stats.upheld, color: '#EF4444' },
          { label: 'Overturned', value: stats.overturned, color: '#10B981' },
        ].map((s) => (
          <Grid item xs={6} md={3} key={s.label}>
            <Paper sx={{ p: 2, bgcolor: '#1E293B', borderRadius: 2, border: '1px solid #334155', textAlign: 'center' }}>
              <Typography variant="h4" fontWeight={700} sx={{ color: s.color }}>{s.value}</Typography>
              <Typography variant="caption" color="#94A3B8">{s.label}</Typography>
            </Paper>
          </Grid>
        ))}
      </Grid>

      <Alert severity="info" sx={{ mb: 2 }}>
        Overturned appeals are logged in the audit trail and can trigger pipeline re-evaluation thresholds adjustment.
      </Alert>

      <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ mb: 2, '& .MuiTab-root': { color: '#94A3B8' }, '& .Mui-selected': { color: '#3B82F6' } }}>
        {['All', 'Pending', 'Upheld', 'Overturned'].map((l) => <Tab key={l} label={l} />)}
      </Tabs>

      {loading ? (
        <Box sx={{ textAlign: 'center', py: 6 }}><CircularProgress /></Box>
      ) : (
        <TableContainer component={Paper} sx={{ bgcolor: '#1E293B', borderRadius: 3, border: '1px solid #334155' }}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Appeal ID</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Claim ID</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Reason</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Status</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Date</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Action</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filtered.map((appeal) => (
                <TableRow key={appeal.appeal_id || appeal.id} hover>
                  <TableCell sx={{ color: '#F1F5F9', fontFamily: 'monospace', fontSize: '0.8rem' }}>{appeal.appeal_id || appeal.id}</TableCell>
                  <TableCell sx={{ color: '#94A3B8', fontFamily: 'monospace' }}>{appeal.claim_id}</TableCell>
                  <TableCell sx={{ color: '#94A3B8', maxWidth: 300 }}>
                    <Typography variant="body2" noWrap>{appeal.reason}</Typography>
                  </TableCell>
                  <TableCell>
                    <Chip label={appeal.status} color={STATUS_COLORS[appeal.status] || 'default'} size="small" sx={{ fontWeight: 600 }} />
                  </TableCell>
                  <TableCell sx={{ color: '#64748B' }}>{appeal.created_at ? new Date(appeal.created_at).toLocaleDateString() : '-'}</TableCell>
                  <TableCell>
                    <Button size="small" startIcon={<Visibility />} onClick={() => setSelected(appeal)}
                      sx={{ color: '#3B82F6', textTransform: 'none' }}>
                      {appeal.status === 'pending' ? 'Review' : 'View'}
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
              {filtered.length === 0 && (
                <TableRow><TableCell colSpan={6} sx={{ textAlign: 'center', py: 4, color: '#64748B' }}>No appeals found</TableCell></TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {/* Review Dialog */}
      <Dialog open={!!selected} onClose={() => setSelected(null)} maxWidth="sm" fullWidth
        PaperProps={{ sx: { bgcolor: '#1E293B', border: '1px solid #334155' } }}>
        <DialogTitle sx={{ color: '#F1F5F9' }}>
          {selected?.status === 'pending' ? 'Review Appeal' : 'Appeal Details'}
        </DialogTitle>
        <DialogContent>
          {selected && (
            <>
              <Alert severity="info" sx={{ mb: 2 }}>Claim: {selected.claim_id}</Alert>
              <Typography variant="subtitle2" color="#94A3B8" gutterBottom>Appeal Reason:</Typography>
              <Paper sx={{ p: 2, mb: 2, bgcolor: '#0F172A', borderRadius: 2 }}>
                <Typography variant="body2" color="#F1F5F9">{selected.reason}</Typography>
              </Paper>

              {selected.status === 'pending' ? (
                <>
                  <Typography variant="subtitle2" color="#F1F5F9" fontWeight={600} gutterBottom>Your Decision</Typography>
                  <RadioGroup value={decision} onChange={(e) => setDecision(e.target.value)}>
                    <FormControlLabel value="upheld" control={<Radio />} label={<Typography color="#F1F5F9">Uphold Original Decision</Typography>} />
                    <FormControlLabel value="overturned" control={<Radio />} label={<Typography color="#F1F5F9">Overturn - Claimant is Right</Typography>} />
                    <FormControlLabel value="partial" control={<Radio />} label={<Typography color="#F1F5F9">Partial - Modify Settlement</Typography>} />
                  </RadioGroup>
                  {decision === 'overturned' && (
                    <Alert severity="warning" sx={{ my: 1 }}>Overturning this decision will be recorded in the audit trail.</Alert>
                  )}
                  <TextField fullWidth multiline rows={3} label="Reasoning *" value={reasoning}
                    onChange={(e) => setReasoning(e.target.value)} sx={{ mt: 2 }}
                    placeholder="Explain your decision..."
                  />
                </>
              ) : (
                <>
                  <Typography variant="subtitle2" color="#94A3B8" gutterBottom>Review Decision:</Typography>
                  <Chip label={selected.status} color={STATUS_COLORS[selected.status]} sx={{ mb: 1 }} />
                  {selected.review_notes && (
                    <Paper sx={{ p: 2, bgcolor: '#0F172A', borderRadius: 2 }}>
                      <Typography variant="body2" color="#94A3B8">{selected.review_notes}</Typography>
                    </Paper>
                  )}
                </>
              )}
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => { setSelected(null); setDecision(''); setReasoning(''); }} sx={{ color: '#94A3B8' }}>Close</Button>
          {selected?.status === 'pending' && (
            <Button variant="contained" onClick={handleReview} disabled={!decision || reasoning.length < 5 || submitting}
              sx={{ borderRadius: 2, background: 'linear-gradient(135deg, #9C27B0, #7B1FA2)' }}>
              {submitting ? <CircularProgress size={20} /> : 'Submit Review'}
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </Box>
  );
}
