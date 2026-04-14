import React, { useEffect, useState } from 'react';
import {
  Box, Paper, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Chip, Button, CircularProgress, IconButton, Tooltip, Alert, Dialog, DialogTitle, DialogContent,
  DialogActions, TextField, MenuItem,
} from '@mui/material';
import { Refresh, Delete, PersonAdd } from '@mui/icons-material';
import { authAPI } from '../../services/api';

const ROLE_COLORS = { user: '#3B82F6', reviewer: '#F59E0B', admin: '#EF4444' };
const ROLE_LABELS = { user: 'Claimant', reviewer: 'Reviewer', admin: 'Admin' };

export default function UserManagement() {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [msg, setMsg] = useState(null);
  const [addOpen, setAddOpen] = useState(false);
  const [form, setForm] = useState({ username: '', email: '', password: '', role: 'user' });

  const fetchUsers = async () => {
    setLoading(true);
    try {
      const res = await authAPI.getAllUsers();
      setUsers(res.data || []);
    } catch {
      setUsers([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchUsers(); }, []);

  const handleDelete = async (id, username) => {
    if (!window.confirm(`Delete user "${username}"? This cannot be undone.`)) return;
    try {
      await authAPI.deleteUser(id);
      setMsg({ type: 'success', text: `Deleted ${username}` });
      fetchUsers();
    } catch (err) {
      setMsg({ type: 'error', text: err.response?.data?.detail || 'Delete failed' });
    }
  };

  const handleAdd = async () => {
    try {
      await authAPI.register(form);
      setMsg({ type: 'success', text: `Created ${form.username}` });
      setAddOpen(false);
      setForm({ username: '', email: '', password: '', role: 'user' });
      fetchUsers();
    } catch (err) {
      setMsg({ type: 'error', text: err.response?.data?.detail || 'Create failed' });
    }
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h5" fontWeight={700} color="#F1F5F9">User Management</Typography>
          <Typography variant="body2" color="#94A3B8">Create, view, and remove user accounts</Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button startIcon={<PersonAdd />} variant="contained" onClick={() => setAddOpen(true)}
            sx={{ borderRadius: 2, background: 'linear-gradient(135deg, #2563EB, #1D4ED8)' }}>
            Add User
          </Button>
          <Button startIcon={<Refresh />} onClick={fetchUsers} sx={{ color: '#94A3B8' }}>Refresh</Button>
        </Box>
      </Box>

      {msg && <Alert severity={msg.type} sx={{ mb: 2 }} onClose={() => setMsg(null)}>{msg.text}</Alert>}

      {loading ? (
        <Box sx={{ textAlign: 'center', py: 6 }}><CircularProgress /></Box>
      ) : (
        <TableContainer component={Paper} sx={{ bgcolor: '#1E293B', borderRadius: 3, border: '1px solid #334155' }}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>ID</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Username</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Email</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Role</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {users.map((u) => (
                <TableRow key={u.id} hover>
                  <TableCell sx={{ color: '#94A3B8', fontFamily: 'monospace' }}>{u.id}</TableCell>
                  <TableCell sx={{ color: '#F1F5F9', fontWeight: 600 }}>{u.username}</TableCell>
                  <TableCell sx={{ color: '#94A3B8' }}>{u.email}</TableCell>
                  <TableCell>
                    <Chip label={ROLE_LABELS[u.role] || u.role} size="small"
                      sx={{ bgcolor: `${ROLE_COLORS[u.role] || '#64748B'}20`, color: ROLE_COLORS[u.role] || '#94A3B8', fontWeight: 600 }} />
                  </TableCell>
                  <TableCell>
                    <Tooltip title="Delete user">
                      <IconButton size="small" sx={{ color: '#EF4444' }} onClick={() => handleDelete(u.id, u.username)}>
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

      {/* Add User Dialog */}
      <Dialog open={addOpen} onClose={() => setAddOpen(false)} maxWidth="sm" fullWidth
        PaperProps={{ sx: { bgcolor: '#0F172A', border: '1px solid #334155', borderRadius: 3 } }}>
        <DialogTitle sx={{ color: '#F1F5F9' }}>Add User</DialogTitle>
        <DialogContent sx={{ pt: 2 }}>
          <TextField fullWidth label="Username" value={form.username} sx={{ mb: 2, mt: 1 }}
            onChange={(e) => setForm({ ...form, username: e.target.value })} />
          <TextField fullWidth label="Email" type="email" value={form.email} sx={{ mb: 2 }}
            onChange={(e) => setForm({ ...form, email: e.target.value })} />
          <TextField fullWidth label="Password" type="password" value={form.password} sx={{ mb: 2 }}
            onChange={(e) => setForm({ ...form, password: e.target.value })} />
          <TextField fullWidth select label="Role" value={form.role} sx={{ mb: 2 }}
            onChange={(e) => setForm({ ...form, role: e.target.value })}>
            <MenuItem value="user">Claimant (user)</MenuItem>
            <MenuItem value="reviewer">Reviewer (HITL approver)</MenuItem>
            <MenuItem value="admin">Admin</MenuItem>
          </TextField>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button onClick={() => setAddOpen(false)} sx={{ color: '#94A3B8' }}>Cancel</Button>
          <Button variant="contained" onClick={handleAdd}
            disabled={!form.username || !form.email || !form.password}
            sx={{ borderRadius: 2, background: 'linear-gradient(135deg, #2563EB, #1D4ED8)' }}>
            Create
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
