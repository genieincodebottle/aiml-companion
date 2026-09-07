import React, { useEffect, useState } from 'react';
import {
  Box, Paper, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Chip, Button, CircularProgress, IconButton, Tooltip, Alert, Dialog, DialogTitle, DialogContent,
  DialogActions, TextField, MenuItem, Grid,
} from '@mui/material';
import { Refresh, Delete, Add, Policy, Edit } from '@mui/icons-material';
import { policyAPI } from '../../services/api';

const TYPE_COLORS = { auto: '#3B82F6', homeowners: '#10B981' };
const STATUS_COLORS = { active: '#10B981', lapsed: '#EF4444', cancelled: '#F59E0B' };

const EMPTY_FORM = {
  policy_number: '', holder_name: '', type: 'auto', status: 'active',
  start_date: '', end_date: '', deductible: 500, premium_monthly: 100,
  claims_count: 0, coverage: '', exclusions: '',
};

export default function PolicyManagement() {
  const [policies, setPolicies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [msg, setMsg] = useState(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [form, setForm] = useState(EMPTY_FORM);
  const [editing, setEditing] = useState(false);

  const fetchPolicies = async () => {
    setLoading(true);
    try {
      const res = await policyAPI.list();
      setPolicies(res.data || []);
    } catch {
      setPolicies([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchPolicies(); }, []);

  const openAdd = () => {
    setForm(EMPTY_FORM);
    setEditing(false);
    setDialogOpen(true);
  };

  const openEdit = (pol) => {
    setForm({
      ...pol,
      coverage: typeof pol.coverage === 'object' ? JSON.stringify(pol.coverage, null, 2) : pol.coverage || '',
      exclusions: Array.isArray(pol.exclusions) ? pol.exclusions.join(', ') : pol.exclusions || '',
    });
    setEditing(true);
    setDialogOpen(true);
  };

  const handleSave = async () => {
    try {
      let coverage = {};
      if (form.coverage) {
        try { coverage = typeof form.coverage === 'string' ? JSON.parse(form.coverage) : form.coverage; }
        catch { setMsg({ type: 'error', text: 'Invalid JSON in coverage field' }); return; }
      }
      const exclusions = typeof form.exclusions === 'string'
        ? form.exclusions.split(',').map((s) => s.trim()).filter(Boolean)
        : form.exclusions || [];

      await policyAPI.upsert({
        ...form,
        deductible: Number(form.deductible) || 0,
        premium_monthly: Number(form.premium_monthly) || 0,
        claims_count: Number(form.claims_count) || 0,
        coverage,
        exclusions,
      });
      setMsg({ type: 'success', text: `${editing ? 'Updated' : 'Created'} policy ${form.policy_number}` });
      setDialogOpen(false);
      fetchPolicies();
    } catch (err) {
      setMsg({ type: 'error', text: err.response?.data?.detail || 'Save failed' });
    }
  };

  const handleDelete = async (num) => {
    if (!window.confirm(`Delete policy "${num}"? This cannot be undone.`)) return;
    try {
      await policyAPI.delete(num);
      setMsg({ type: 'success', text: `Deleted ${num}` });
      fetchPolicies();
    } catch (err) {
      setMsg({ type: 'error', text: err.response?.data?.detail || 'Delete failed' });
    }
  };

  const f = (k, v) => setForm((prev) => ({ ...prev, [k]: v }));

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h5" fontWeight={700} color="#F1F5F9">Policy Management</Typography>
          <Typography variant="body2" color="#94A3B8">Seed and manage policies used by the claims pipeline</Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button startIcon={<Add />} variant="contained" onClick={openAdd}
            sx={{ borderRadius: 2, background: 'linear-gradient(135deg, #2563EB, #1D4ED8)' }}>
            Add Policy
          </Button>
          <Button startIcon={<Refresh />} onClick={fetchPolicies} sx={{ color: '#94A3B8' }}>Refresh</Button>
        </Box>
      </Box>

      {msg && <Alert severity={msg.type} sx={{ mb: 2 }} onClose={() => setMsg(null)}>{msg.text}</Alert>}

      {loading ? (
        <Box sx={{ textAlign: 'center', py: 6 }}><CircularProgress /></Box>
      ) : policies.length === 0 ? (
        <Paper sx={{ p: 4, textAlign: 'center', bgcolor: '#1E293B', border: '1px solid #334155', borderRadius: 3 }}>
          <Policy sx={{ fontSize: 48, color: '#334155', mb: 1 }} />
          <Typography color="#94A3B8">No policies found. Add one or run <code>python scripts/seed_policies.py</code></Typography>
        </Paper>
      ) : (
        <TableContainer component={Paper} sx={{ bgcolor: '#1E293B', borderRadius: 3, border: '1px solid #334155' }}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Policy Number</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Holder</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Type</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Status</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Period</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Deductible</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {policies.map((p) => (
                <TableRow key={p.policy_number} hover>
                  <TableCell sx={{ color: '#F1F5F9', fontFamily: 'monospace', fontWeight: 600 }}>{p.policy_number}</TableCell>
                  <TableCell sx={{ color: '#94A3B8' }}>{p.holder_name}</TableCell>
                  <TableCell>
                    <Chip label={p.type} size="small"
                      sx={{ bgcolor: `${TYPE_COLORS[p.type] || '#64748B'}20`, color: TYPE_COLORS[p.type] || '#94A3B8', fontWeight: 600 }} />
                  </TableCell>
                  <TableCell>
                    <Chip label={p.status} size="small"
                      sx={{ bgcolor: `${STATUS_COLORS[p.status] || '#64748B'}20`, color: STATUS_COLORS[p.status] || '#94A3B8', fontWeight: 600 }} />
                  </TableCell>
                  <TableCell sx={{ color: '#94A3B8', fontSize: '0.8rem' }}>{p.start_date} to {p.end_date}</TableCell>
                  <TableCell sx={{ color: '#94A3B8' }}>{p.deductible?.toLocaleString()}</TableCell>
                  <TableCell>
                    <Tooltip title="Edit policy">
                      <IconButton size="small" sx={{ color: '#3B82F6' }} onClick={() => openEdit(p)}>
                        <Edit fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Delete policy">
                      <IconButton size="small" sx={{ color: '#EF4444' }} onClick={() => handleDelete(p.policy_number)}>
                        <Delete fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {/* Add / Edit Dialog */}
      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="md" fullWidth
        PaperProps={{ sx: { bgcolor: '#0F172A', border: '1px solid #334155', borderRadius: 3 } }}>
        <DialogTitle sx={{ color: '#F1F5F9' }}>{editing ? 'Edit Policy' : 'Add Policy'}</DialogTitle>
        <DialogContent sx={{ pt: 2 }}>
          <Grid container spacing={2} sx={{ mt: 0.5 }}>
            <Grid item xs={6}>
              <TextField fullWidth label="Policy Number" value={form.policy_number}
                disabled={editing} onChange={(e) => f('policy_number', e.target.value)} />
            </Grid>
            <Grid item xs={6}>
              <TextField fullWidth label="Holder Name" value={form.holder_name}
                onChange={(e) => f('holder_name', e.target.value)} />
            </Grid>
            <Grid item xs={4}>
              <TextField fullWidth select label="Type" value={form.type}
                onChange={(e) => f('type', e.target.value)}>
                <MenuItem value="auto">Auto</MenuItem>
                <MenuItem value="homeowners">Homeowners</MenuItem>
              </TextField>
            </Grid>
            <Grid item xs={4}>
              <TextField fullWidth select label="Status" value={form.status}
                onChange={(e) => f('status', e.target.value)}>
                <MenuItem value="active">Active</MenuItem>
                <MenuItem value="lapsed">Lapsed</MenuItem>
                <MenuItem value="cancelled">Cancelled</MenuItem>
              </TextField>
            </Grid>
            <Grid item xs={4}>
              <TextField fullWidth label="Deductible" type="number" value={form.deductible}
                onChange={(e) => f('deductible', e.target.value)} />
            </Grid>
            <Grid item xs={4}>
              <TextField fullWidth label="Start Date" type="date" value={form.start_date}
                InputLabelProps={{ shrink: true }} onChange={(e) => f('start_date', e.target.value)} />
            </Grid>
            <Grid item xs={4}>
              <TextField fullWidth label="End Date" type="date" value={form.end_date}
                InputLabelProps={{ shrink: true }} onChange={(e) => f('end_date', e.target.value)} />
            </Grid>
            <Grid item xs={4}>
              <TextField fullWidth label="Monthly Premium" type="number" value={form.premium_monthly}
                onChange={(e) => f('premium_monthly', e.target.value)} />
            </Grid>
            <Grid item xs={12}>
              <TextField fullWidth label="Coverage (JSON)" multiline rows={4} value={form.coverage}
                placeholder='{"collision": 50000, "comprehensive": 50000, "liability": 100000}'
                onChange={(e) => f('coverage', e.target.value)}
                helperText="JSON object mapping coverage type to limit amount" />
            </Grid>
            <Grid item xs={12}>
              <TextField fullWidth label="Exclusions (comma-separated)" value={form.exclusions}
                placeholder="racing, commercial_use, pre_existing_damage"
                onChange={(e) => f('exclusions', e.target.value)} />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button onClick={() => setDialogOpen(false)} sx={{ color: '#94A3B8' }}>Cancel</Button>
          <Button variant="contained" onClick={handleSave}
            disabled={!form.policy_number || !form.holder_name || !form.start_date || !form.end_date}
            sx={{ borderRadius: 2, background: 'linear-gradient(135deg, #2563EB, #1D4ED8)' }}>
            {editing ? 'Update' : 'Create'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
