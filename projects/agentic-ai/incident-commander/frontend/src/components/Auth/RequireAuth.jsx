import { Navigate, useLocation } from 'react-router-dom';
import { Box, CircularProgress } from '@mui/material';
import { useAuthStore } from '../../store/authStore';

// Route guard.
//
// Two properties worth stating, because both are easy to get subtly wrong:
//
// 1. It waits for `initializing`. Redirecting while the session is still being
//    verified bounces an authenticated user to /login on every hard refresh -
//    the token is in storage but `user` has not arrived yet.
// 2. It is a *convenience*, not a control. Anyone can edit the bundle and render
//    whatever they like; what actually protects an approval is `require_role` on
//    the server. This only spares an unauthenticated user a screen of 401s.
export default function RequireAuth({ children }) {
  const location = useLocation();
  const { user, loginRequired, initializing } = useAuthStore();

  if (initializing) {
    return (
      <Box sx={{ minHeight: '60vh', display: 'grid', placeItems: 'center' }}>
        <CircularProgress />
      </Box>
    );
  }

  if (loginRequired && !user) {
    // Remember where they were headed so login can put them back there.
    return <Navigate to="/login" replace state={{ from: location.pathname + location.search }} />;
  }

  return children;
}
