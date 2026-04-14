import React, { useEffect, useState } from 'react';
import {
  Box, Paper, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Chip, Button, CircularProgress, Dialog, DialogTitle, DialogContent, DialogActions,
} from '@mui/material';
import { HowToVote, Refresh, Visibility } from '@mui/icons-material';
import { appealsAPI } from '../../services/api';
import useAuthStore from '../../store/authStore';

const STATUS_COLORS = { pending: 'warning', upheld: 'error', overturned: 'success', partial: 'info' };

export default function UserAppeals() {
  const user = useAuthStore((s) => s.user);
  const [appeals, setAppeals] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    const fetch = async () => {
      try {
        const res = await appealsAPI.getUserAppeals(user?.id || user?.username);
        setAppeals(res.data.appeals || res.data || []);
      } catch { setAppeals([]); }
      finally { setLoading(false); }
    };
    fetch();
  }, []);

  return (
    <Box>
      <Typography variant="h5" fontWeight={700} color="#F1F5F9" gutterBottom>
        <HowToVote sx={{ mr: 1, verticalAlign: 'middle', color: '#F59E0B' }} />
        My Appeals
      </Typography>
      <Typography variant="body2" color="#94A3B8" sx={{ mb: 3 }}>Track appeals you have submitted against claim decisions</Typography>

      {loading ? (
        <Box sx={{ textAlign: 'center', py: 6 }}><CircularProgress /></Box>
      ) : appeals.length === 0 ? (
        <Paper sx={{ p: 6, bgcolor: '#1E293B', borderRadius: 3, textAlign: 'center', border: '1px solid #334155' }}>
          <HowToVote sx={{ fontSize: 64, color: '#334155', mb: 2 }} />
          <Typography variant="h6" color="#94A3B8">No appeals yet</Typography>
          <Typography variant="body2" color="#64748B">You can appeal denied or partially approved claims from the My Claims page</Typography>
        </Paper>
      ) : (
        <TableContainer component={Paper} sx={{ bgcolor: '#1E293B', borderRadius: 3, border: '1px solid #334155' }}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Appeal ID</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Claim ID</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Reason</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Status</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Submitted</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Action</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {appeals.map((a) => (
                <TableRow key={a.appeal_id || a.id} hover>
                  <TableCell sx={{ color: '#F1F5F9', fontFamily: 'monospace' }}>{a.appeal_id || a.id}</TableCell>
                  <TableCell sx={{ color: '#94A3B8', fontFamily: 'monospace' }}>{a.claim_id}</TableCell>
                  <TableCell sx={{ color: '#94A3B8', maxWidth: 250 }}><Typography variant="body2" noWrap>{a.reason}</Typography></TableCell>
                  <TableCell><Chip label={a.status} color={STATUS_COLORS[a.status] || 'default'} size="small" /></TableCell>
                  <TableCell sx={{ color: '#64748B' }}>{a.created_at ? new Date(a.created_at).toLocaleDateString() : '-'}</TableCell>
                  <TableCell>
                    <Button size="small" startIcon={<Visibility />} onClick={() => setSelected(a)} sx={{ color: '#3B82F6', textTransform: 'none' }}>View</Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      <Dialog open={!!selected} onClose={() => setSelected(null)} maxWidth="sm" fullWidth
        PaperProps={{ sx: { bgcolor: '#1E293B', border: '1px solid #334155' } }}>
        <DialogTitle sx={{ color: '#F1F5F9' }}>Appeal Details</DialogTitle>
        <DialogContent>
          {selected && (
            <>
              <Chip label={selected.status} color={STATUS_COLORS[selected.status]} sx={{ mb: 2 }} />
              <Typography variant="subtitle2" color="#94A3B8" gutterBottom>Claim: {selected.claim_id}</Typography>
              <Typography variant="subtitle2" color="#94A3B8" sx={{ mt: 2 }} gutterBottom>Your Appeal Reason:</Typography>
              <Paper sx={{ p: 2, bgcolor: '#0F172A', borderRadius: 2, mb: 2 }}>
                <Typography variant="body2" color="#F1F5F9">{selected.reason}</Typography>
              </Paper>
              {selected.review_notes && (
                <>
                  <Typography variant="subtitle2" color="#94A3B8" gutterBottom>Reviewer Notes:</Typography>
                  <Paper sx={{ p: 2, bgcolor: '#0F172A', borderRadius: 2 }}>
                    <Typography variant="body2" color="#94A3B8">{selected.review_notes}</Typography>
                  </Paper>
                </>
              )}
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSelected(null)} sx={{ color: '#94A3B8' }}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
