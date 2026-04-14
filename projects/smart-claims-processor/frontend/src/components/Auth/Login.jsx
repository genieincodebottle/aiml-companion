import React, { useState } from 'react';
import {
  Box, Paper, Typography, TextField, Button, Tab, Tabs, Alert,
  InputAdornment, IconButton, CircularProgress,
} from '@mui/material';
import {
  Visibility, VisibilityOff, AccountBalance, Person, Lock, Email,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { authAPI } from '../../services/api';
import useAuthStore from '../../store/authStore';

export default function Login() {
  const [tab, setTab] = useState(0);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [email, setEmail] = useState('');
  const [role, setRole] = useState('user');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const login = useAuthStore((s) => s.login);

  const landingFor = (r) => (r === 'user' ? '/my-claims' : '/dashboard');

  // Kill the Chrome/Safari autofill blue background on a dark theme.
  const fieldSx = {
    mb: 2,
    '& input:-webkit-autofill, & input:-webkit-autofill:hover, & input:-webkit-autofill:focus, & input:-webkit-autofill:active': {
      WebkitBoxShadow: '0 0 0 100px #1E293B inset',
      WebkitTextFillColor: '#F1F5F9',
      caretColor: '#F1F5F9',
      transition: 'background-color 5000s ease-in-out 0s',
    },
  };

  const handleLogin = async (e) => {
    e?.preventDefault();
    setError('');
    setLoading(true);
    try {
      const res = await authAPI.login(username, password);
      login(res.data.user, res.data.token);
      navigate(landingFor(res.data.user.role));
    } catch (err) {
      setError(err.response?.data?.detail || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  const handleRegister = async (e) => {
    e?.preventDefault();
    setError('');
    setLoading(true);
    try {
      const res = await authAPI.register({ username, password, email, role });
      login(res.data.user, res.data.token);
      navigate(landingFor(res.data.user.role));
    } catch (err) {
      setError(err.response?.data?.detail || 'Registration failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%)' }}>
      <Paper sx={{ width: 480, maxWidth: '95vw', p: 4, borderRadius: 3, bgcolor: '#1E293B', border: '1px solid #334155' }}>
        {/* Header */}
        <Box sx={{ textAlign: 'center', mb: 3 }}>
          <AccountBalance sx={{ fontSize: 48, color: '#3B82F6', mb: 1 }} />
          <Typography variant="h5" fontWeight={700} color="#F1F5F9">Smart Claims Processor</Typography>
          <Typography variant="body2" color="#94A3B8" sx={{ mt: 0.5 }}>AI-Powered Insurance Claims Processing</Typography>
        </Box>

        {/* Tabs */}
        <Tabs value={tab} onChange={(_, v) => { setTab(v); setError(''); }} centered sx={{ mb: 2, '& .MuiTab-root': { color: '#94A3B8', fontWeight: 600 }, '& .Mui-selected': { color: '#3B82F6' } }}>
          <Tab label="Sign In" />
          <Tab label="Register" />
        </Tabs>

        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

        <form onSubmit={tab === 0 ? handleLogin : handleRegister}>
          <TextField
            fullWidth label="Username" value={username}
            onChange={(e) => setUsername(e.target.value)}
            sx={fieldSx}
            InputProps={{ startAdornment: <InputAdornment position="start"><Person sx={{ color: '#64748B' }} /></InputAdornment> }}
          />
          {tab === 1 && (
            <TextField
              fullWidth label="Email" type="email" value={email}
              onChange={(e) => setEmail(e.target.value)}
              sx={fieldSx}
              InputProps={{ startAdornment: <InputAdornment position="start"><Email sx={{ color: '#64748B' }} /></InputAdornment> }}
            />
          )}
          <TextField
            fullWidth label="Password" type={showPassword ? 'text' : 'password'}
            value={password} onChange={(e) => setPassword(e.target.value)}
            sx={fieldSx}
            InputProps={{
              startAdornment: <InputAdornment position="start"><Lock sx={{ color: '#64748B' }} /></InputAdornment>,
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton onClick={() => setShowPassword(!showPassword)} edge="end" sx={{ color: '#64748B' }}>
                    {showPassword ? <VisibilityOff /> : <Visibility />}
                  </IconButton>
                </InputAdornment>
              ),
            }}
          />
          {tab === 1 && (
            <TextField
              fullWidth select label="Role" value={role}
              onChange={(e) => setRole(e.target.value)}
              sx={fieldSx}
              SelectProps={{ native: true }}
            >
              <option value="user">Claimant (user)</option>
              <option value="reviewer">Reviewer (HITL approver)</option>
            </TextField>
          )}
          <Button
            fullWidth type="submit" variant="contained" size="large" disabled={loading || !username || !password}
            sx={{ py: 1.5, fontWeight: 600, fontSize: '1rem', borderRadius: 2, background: 'linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%)' }}
          >
            {loading ? <CircularProgress size={24} /> : tab === 0 ? 'Sign In' : 'Create Account'}
          </Button>
        </form>
      </Paper>
    </Box>
  );
}
