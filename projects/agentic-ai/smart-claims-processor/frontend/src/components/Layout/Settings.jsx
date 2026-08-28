import React, { useState } from 'react';
import {
  Box, Paper, Typography, TextField, Button, Grid, Alert, Chip, Divider, CircularProgress,
} from '@mui/material';
import { Settings as SettingsIcon } from '@mui/icons-material';
import useAuthStore from '../../store/authStore';
import { authAPI } from '../../services/api';

// Must match backend roles: user | reviewer | admin
const ROLE_COLORS = {
  user: '#3B82F6',
  reviewer: '#F59E0B',
  admin: '#EF4444',
};

const ROLE_LABELS = {
  user: 'Claimant',
  reviewer: 'Reviewer',
  admin: 'Admin',
};

export default function Settings() {
  const { user, updateUser } = useAuthStore();
  const [currentPass, setCurrentPass] = useState('');
  const [newPass, setNewPass] = useState('');
  const [success, setSuccess] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handlePasswordChange = async () => {
    setError('');
    setSuccess('');
    setLoading(true);
    try {
      await authAPI.updatePassword(currentPass, newPass);
      setSuccess('Password updated successfully');
      setCurrentPass('');
      setNewPass('');
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to update password');
    } finally {
      setLoading(false);
    }
  };

  const roleColor = ROLE_COLORS[user?.role] || '#64748B';

  return (
    <Box>
      <Typography variant="h5" fontWeight={700} color="#F1F5F9" gutterBottom>
        <SettingsIcon sx={{ mr: 1, verticalAlign: 'middle', color: '#94A3B8' }} />
        Account Settings
      </Typography>

      <Grid container spacing={3} sx={{ mt: 1 }}>
        {/* Profile */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, bgcolor: '#1E293B', borderRadius: 3, border: '1px solid #334155' }}>
            <Typography variant="subtitle1" fontWeight={600} color="#F1F5F9" gutterBottom>Profile</Typography>
            <Divider sx={{ borderColor: '#334155', mb: 2 }} />
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Typography variant="caption" color="#64748B">Username</Typography>
                <Typography color="#F1F5F9" fontWeight={600}>{user?.username}</Typography>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="caption" color="#64748B">Email</Typography>
                <Typography color="#94A3B8">{user?.email || 'Not set'}</Typography>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="caption" color="#64748B">Role</Typography>
                <Box sx={{ mt: 0.5 }}>
                  <Chip label={ROLE_LABELS[user?.role] || user?.role} sx={{ bgcolor: `${roleColor}20`, color: roleColor, fontWeight: 600, border: `1px solid ${roleColor}40` }} />
                </Box>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* Password */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, bgcolor: '#1E293B', borderRadius: 3, border: '1px solid #334155' }}>
            <Typography variant="subtitle1" fontWeight={600} color="#F1F5F9" gutterBottom>Change Password</Typography>
            <Divider sx={{ borderColor: '#334155', mb: 2 }} />
            {success && <Alert severity="success" sx={{ mb: 2 }}>{success}</Alert>}
            {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
            <TextField fullWidth label="Current Password" type="password" value={currentPass}
              onChange={(e) => setCurrentPass(e.target.value)} sx={{ mb: 2 }} />
            <TextField fullWidth label="New Password" type="password" value={newPass}
              onChange={(e) => setNewPass(e.target.value)} sx={{ mb: 2 }} helperText="Minimum 6 characters" />
            <Button variant="contained" onClick={handlePasswordChange}
              disabled={!currentPass || newPass.length < 6 || loading}
              sx={{ borderRadius: 2, background: 'linear-gradient(135deg, #2563EB, #1D4ED8)' }}>
              {loading ? <CircularProgress size={20} /> : 'Update Password'}
            </Button>
          </Paper>
        </Grid>

        {/* Role Permissions Reference */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3, bgcolor: '#1E293B', borderRadius: 3, border: '1px solid #334155' }}>
            <Typography variant="subtitle1" fontWeight={600} color="#F1F5F9" gutterBottom>Role Permissions</Typography>
            <Divider sx={{ borderColor: '#334155', mb: 2 }} />
            <Grid container spacing={2}>
              {[
                { role: 'user', label: 'Claimant', perms: ['Submit claims', 'View own claims', 'Track claim status', 'Submit appeals'] },
                { role: 'reviewer', label: 'Reviewer', perms: ['Claims dashboard (all claims)', 'HITL review queue (approve/deny)', 'Appeals review', 'Analytics dashboard'] },
                { role: 'admin', label: 'Admin', perms: ['All reviewer permissions', 'User management', 'Switch LLM provider (Gemini/Groq)', 'Full system access'] },
              ].map((r) => (
                <Grid item xs={12} md={4} key={r.role}>
                  <Paper sx={{ p: 2, bgcolor: user?.role === r.role ? `${ROLE_COLORS[r.role]}10` : '#0F172A', borderRadius: 2, border: `1px solid ${user?.role === r.role ? ROLE_COLORS[r.role] : '#334155'}` }}>
                    <Chip label={r.label} size="small" sx={{ mb: 1, bgcolor: `${ROLE_COLORS[r.role]}20`, color: ROLE_COLORS[r.role], fontWeight: 600 }} />
                    {r.perms.map((p) => (
                      <Typography key={p} variant="body2" color="#94A3B8" sx={{ fontSize: '0.8rem' }}>- {p}</Typography>
                    ))}
                  </Paper>
                </Grid>
              ))}
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}
