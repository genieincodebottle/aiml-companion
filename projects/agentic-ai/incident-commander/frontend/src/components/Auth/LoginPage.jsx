import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Container,
  Divider,
  IconButton,
  InputAdornment,
  Stack,
  TextField,
  Typography,
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { Bolt, Visibility, VisibilityOff, LockOutlined } from '@mui/icons-material';
import { useAuthStore } from '../../store/authStore';
import { GRADIENTS } from '../../theme';
import LoopAnimation from '../Flow/LoopAnimation';

// Split login: brand and product on the left over a gradient, the form on a
// floating card to the right.
//
// The left panel is not decoration. Someone who reaches a login screen for an
// unfamiliar tool has no idea what they are about to log into, and a bare
// username/password box answers none of that - so the panel carries the same
// one-line pitch and the same live loop diagram the home page uses.
export default function LoginPage() {
  const navigate = useNavigate();
  const theme = useTheme();
  const { login, loggingIn, error, user, loginRequired, initializing } = useAuthStore();

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);

  // Always home after signing in.
  //
  // RequireAuth still records the attempted route in `location.state.from`, and
  // returning there is the usual pattern - but it is the wrong default here. A
  // session expiring mid-incident would drop you back onto a console for a run
  // that is no longer streaming. Home orients you and is one click from anywhere.
  // To restore deep-linking, read `location.state?.from` instead.
  const LANDING = '/home';

  // Already signed in, or the backend does not want a login at all: get out of
  // the way rather than making someone authenticate for no reason.
  useEffect(() => {
    if (!initializing && (user || !loginRequired)) navigate(LANDING, { replace: true });
  }, [user, loginRequired, initializing, navigate]);

  const submit = async (e) => {
    e.preventDefault();
    if (await login(username.trim(), password)) navigate(LANDING, { replace: true });
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        p: { xs: 2, md: 4 },
        background: GRADIENTS.page(theme.palette.mode),
      }}
    >
      <Container maxWidth="lg">
        <Card
          sx={{
            overflow: 'hidden',
            borderRadius: 4,
            boxShadow:
              theme.palette.mode === 'light'
                ? '0 24px 60px rgba(15,23,42,0.18)'
                : '0 24px 60px rgba(0,0,0,0.55)',
          }}
        >
          <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' } }}>
            {/* --- brand panel --- */}
            <Box
              sx={{
                flex: { md: '1 1 52%' },
                p: { xs: 3, md: 5 },
                color: '#f8fafc',
                background: GRADIENTS.brand,
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.25, mb: 2 }}>
                <Bolt sx={{ fontSize: 34 }} />
                <Typography variant="h5" sx={{ fontWeight: 800 }}>
                  Incident Commander
                </Typography>
              </Box>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 1, opacity: 0.95 }}>
                An autonomous SRE that knows when to stop
              </Typography>
              <Typography variant="body2" sx={{ opacity: 0.85, mb: 2 }}>
                Seven agents diagnose and fix a production incident inside a bounded loop. They
                argue with each other, stop for your approval, and escalate rather than guess.
              </Typography>

              <Box sx={{ display: { xs: 'none', md: 'block' }, mx: -2 }}>
                <LoopAnimation height={260} interactive={false} activeIndex={null} onBrand />
              </Box>

              <Stack direction="row" spacing={1} sx={{ mt: 2, flexWrap: 'wrap', gap: 1 }}>
                {['7 agents', '6 halt conditions', 'human-gated', 'audited'].map((t) => (
                  <Chip
                    key={t}
                    size="small"
                    label={t}
                    sx={{ bgcolor: 'rgba(255,255,255,0.18)', color: '#f8fafc', fontWeight: 600 }}
                  />
                ))}
              </Stack>
            </Box>

            {/* --- form --- */}
            <Box sx={{ flex: { md: '1 1 48%' }, display: 'flex', alignItems: 'center' }}>
              <CardContent sx={{ p: { xs: 3, md: 5 }, width: '100%' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                  <LockOutlined color="primary" />
                  <Typography variant="h5" sx={{ fontWeight: 700 }}>
                    Sign in
                  </Typography>
                </Box>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                  This console can approve a production rollback, so it asks who you are.
                </Typography>

                {error && (
                  <Alert severity="error" variant="outlined" sx={{ mb: 2 }}>
                    {error}
                  </Alert>
                )}

                <form onSubmit={submit}>
                  <TextField
                    fullWidth
                    required
                    autoFocus
                    autoComplete="username"
                    label="Username"
                    margin="normal"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                  />
                  <TextField
                    fullWidth
                    required
                    autoComplete="current-password"
                    label="Password"
                    margin="normal"
                    type={showPassword ? 'text' : 'password'}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    slotProps={{
                      input: {
                        endAdornment: (
                          <InputAdornment position="end">
                            <IconButton
                              edge="end"
                              aria-label={showPassword ? 'Hide password' : 'Show password'}
                              onClick={() => setShowPassword((v) => !v)}
                            >
                              {showPassword ? <VisibilityOff /> : <Visibility />}
                            </IconButton>
                          </InputAdornment>
                        ),
                      },
                    }}
                  />
                  <Button
                    type="submit"
                    fullWidth
                    size="large"
                    variant="contained"
                    disabled={loggingIn || !username || !password}
                    startIcon={loggingIn ? <CircularProgress size={18} color="inherit" /> : null}
                    sx={{
                      mt: 3,
                      py: 1.4,
                      background: GRADIENTS.brand,
                      '&:hover': { background: GRADIENTS.brandHover },
                    }}
                  >
                    {loggingIn ? 'Signing in...' : 'Sign in'}
                  </Button>
                </form>

                <Divider sx={{ my: 3 }} />
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                  Accounts are created by an admin - there is deliberately no self-service signup,
                  because this console can approve a production rollback. A fresh database seeds one
                  admin from
                  <code> IC_SEED_ADMIN_USER</code> / <code>IC_SEED_ADMIN_PASSWORD</code>.
                </Typography>
              </CardContent>
            </Box>
          </Box>
        </Card>
      </Container>
    </Box>
  );
}
