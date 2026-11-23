import { useEffect } from 'react';
import { Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import Layout from './components/Layout/Layout';
import IncidentConsole from './components/Console/IncidentConsole';
import IncidentHistory from './components/History/IncidentHistory';
import WorkflowView from './components/Flow/WorkflowView';
import HomePage from './components/Home/HomePage';
import LoginPage from './components/Auth/LoginPage';
import RequireAuth from './components/Auth/RequireAuth';
import { useAuthStore } from './store/authStore';
import { setUnauthorizedHandler } from './services/api';

export default function App() {
  const navigate = useNavigate();
  const bootstrap = useAuthStore((s) => s.bootstrap);
  const clearSession = useAuthStore((s) => s.clearSession);

  // Ask the server who we are before rendering anything that assumes a session.
  useEffect(() => {
    bootstrap();
  }, [bootstrap]);

  // An expired token surfaces as a 401 on some ordinary request. Route to /login
  // from here rather than from the interceptor, so it is a React navigation and
  // not a full page reload that discards in-flight state.
  useEffect(() => {
    setUnauthorizedHandler(() => {
      clearSession();
      navigate('/login', { replace: true });
    });
    return () => setUnauthorizedHandler(null);
  }, [navigate, clearSession]);

  return (
    <Routes>
      {/* Login is outside the Layout: no sidebar, no app chrome. */}
      <Route path="/login" element={<LoginPage />} />

      <Route
        path="*"
        element={
          <RequireAuth>
            <Layout>
              <Routes>
                <Route path="/home" element={<HomePage />} />
                <Route path="/console" element={<IncidentConsole />} />
                <Route path="/flow" element={<WorkflowView />} />
                <Route path="/history" element={<IncidentHistory />} />
                <Route path="/" element={<Navigate to="/home" replace />} />
                <Route path="*" element={<Navigate to="/home" replace />} />
              </Routes>
            </Layout>
          </RequireAuth>
        }
      />
    </Routes>
  );
}
