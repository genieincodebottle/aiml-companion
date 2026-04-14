import React, { useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import useAuthStore from './store/authStore';
import { loadCurrency } from './services/currency';

// Components
import Login from './components/Auth/Login';
import Layout from './components/Layout/Layout';
import MyClaims from './components/Claims/MyClaims';
import SubmitClaim from './components/Claims/SubmitClaim';
import UserAppeals from './components/Claims/UserAppeals';
import ClaimsDashboard from './components/Dashboard/ClaimsDashboard';
import HITLQueue from './components/HITL/HITLQueue';
import AppealsManagement from './components/HITL/AppealsManagement';
import AnalyticsDashboard from './components/Analytics/AnalyticsDashboard';
import UserManagement from './components/Auth/UserManagement';
import PolicyManagement from './components/Admin/PolicyManagement';
import Settings from './components/Layout/Settings';

// Dark theme matching the Streamlit dashboard
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#3B82F6' },
    secondary: { main: '#8B5CF6' },
    success: { main: '#10B981' },
    warning: { main: '#F59E0B' },
    error: { main: '#EF4444' },
    info: { main: '#0EA5E9' },
    background: { default: '#0F172A', paper: '#1E293B' },
    text: { primary: '#F1F5F9', secondary: '#94A3B8' },
    divider: '#334155',
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
  },
  shape: { borderRadius: 8 },
  components: {
    MuiPaper: { styleOverrides: { root: { backgroundImage: 'none' } } },
    MuiCard: { styleOverrides: { root: { borderRadius: 12 } } },
    MuiButton: { styleOverrides: { root: { textTransform: 'none', fontWeight: 600 } } },
    MuiChip: { styleOverrides: { root: { fontWeight: 500 } } },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 8,
            '& fieldset': { borderColor: '#334155' },
            '&:hover fieldset': { borderColor: '#475569' },
          },
          '& .MuiInputLabel-root': { color: '#64748B' },
          '& .MuiInputBase-input': { color: '#F1F5F9' },
        },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        root: { borderColor: '#334155', padding: '12px 16px' },
        head: { backgroundColor: '#0F172A' },
      },
    },
    MuiTableRow: {
      styleOverrides: {
        root: { '&:hover': { backgroundColor: '#1E293B80' } },
      },
    },
    MuiDialog: {
      styleOverrides: {
        paper: { borderRadius: 12 },
      },
    },
  },
});

// Pick the correct landing for the current user's role.
// Claimants (role=user) see My Claims; reviewers/admins see the staff dashboard.
function landingFor(user) {
  if (!user) return '/login';
  return user.role === 'user' ? '/my-claims' : '/dashboard';
}

// Protected route wrapper
function ProtectedRoute({ children, minRole }) {
  const { isAuthenticated, hasRole, user } = useAuthStore();
  if (!isAuthenticated) return <Navigate to="/login" replace />;
  if (minRole && !hasRole(minRole)) return <Navigate to={landingFor(user)} replace />;
  return children;
}

export default function App() {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const user = useAuthStore((s) => s.user);
  const landing = landingFor(user);

  // Load currency symbol from the active country profile on mount.
  useEffect(() => { if (isAuthenticated) loadCurrency(); }, [isAuthenticated]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>
        <Routes>
          {/* Public */}
          <Route path="/login" element={isAuthenticated ? <Navigate to={landing} /> : <Login />} />

          {/* Authenticated - wrapped in Layout */}
          <Route path="/" element={<ProtectedRoute><Layout /></ProtectedRoute>}>
            {/* Default landing depends on role */}
            <Route index element={<Navigate to={landing} replace />} />

            {/* Claimant (role=user) */}
            <Route path="my-claims" element={<MyClaims />} />
            <Route path="submit-claim" element={<SubmitClaim />} />
            <Route path="my-appeals" element={<UserAppeals />} />

            {/* Everyone */}
            <Route path="settings" element={<Settings />} />

            {/* Reviewer + Admin (role >= reviewer) */}
            <Route path="dashboard"  element={<ProtectedRoute minRole="reviewer"><ClaimsDashboard /></ProtectedRoute>} />
            <Route path="hitl-queue" element={<ProtectedRoute minRole="reviewer"><HITLQueue /></ProtectedRoute>} />
            <Route path="appeals"    element={<ProtectedRoute minRole="reviewer"><AppealsManagement /></ProtectedRoute>} />
            <Route path="analytics"  element={<ProtectedRoute minRole="reviewer"><AnalyticsDashboard /></ProtectedRoute>} />

            {/* Admin-only */}
            <Route path="user-management" element={<ProtectedRoute minRole="admin"><UserManagement /></ProtectedRoute>} />
            <Route path="policy-management" element={<ProtectedRoute minRole="admin"><PolicyManagement /></ProtectedRoute>} />
          </Route>

          {/* Catch all */}
          <Route path="*" element={<Navigate to={isAuthenticated ? landing : '/login'} replace />} />
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
}
