import { create } from 'zustand';
import { authAPI, getToken, setToken } from '../services/api';

// Session state.
//
// The important behaviour is `bootstrap()`. A token in localStorage is a *claim*,
// not proof: it may be expired, signed with a rotated secret, or belong to a user
// that has since been deactivated. So on boot the app asks the server who it is
// rather than trusting the stored blob and rendering a logged-in shell that 401s
// on its first real request.
export const useAuthStore = create((set, get) => ({
  user: null,
  //: Whether the backend actually requires a login. When false the whole auth UI
  //: stays out of the way, which is the default zero-infrastructure setup.
  loginRequired: false,
  //: True until bootstrap finishes. Routes must wait rather than briefly
  //: redirecting an authenticated user to /login on every refresh.
  initializing: true,
  loggingIn: false,
  error: null,

  bootstrap: async () => {
    set({ initializing: true });
    try {
      const data = await authAPI.me();
      set({
        user: data.user,
        loginRequired: Boolean(data.login_required),
        initializing: false,
        error: null,
      });
    } catch (err) {
      if (err?.response?.status === 401) {
        // The server saw the token and rejected it - expired, rotated secret,
        // deactivated user. That is the only case where dropping it is right,
        // and it also means auth is switched on, so the login page must show.
        setToken(null);
        set({ user: null, loginRequired: true, initializing: false });
      } else {
        // No usable response (backend restarting, network blip): the token may
        // be perfectly valid, so keep it and whatever we last knew about
        // loginRequired. Deleting it here silently logged people out on every
        // backend redeploy.
        set({ initializing: false });
      }
    }
  },

  login: async (username, password) => {
    set({ loggingIn: true, error: null });
    try {
      const data = await authAPI.login(username, password);
      setToken(data.access_token);
      set({ user: data.user, loggingIn: false, loginRequired: true });
      return true;
    } catch (err) {
      set({
        loggingIn: false,
        error: err?.response?.data?.detail || err?.message || 'Login failed',
      });
      return false;
    }
  },

  logout: async () => {
    // Best-effort server call. The token is dropped locally even if the POST
    // fails (authAPI.logout clears it in a finally), so the client must finish
    // logging out too - a network blip mid-logout used to leave the rejection
    // unhandled and the UI still showing a signed-in user.
    try {
      await authAPI.logout();
    } catch {
      // Server unreachable; nothing to undo, the session is already gone locally.
    } finally {
      set({ user: null });
    }
  },

  // Called by the axios 401 interceptor.
  clearSession: () => set({ user: null }),

  hasToken: () => Boolean(getToken()),

  // Viewers can watch; operators and admins can approve and cancel. Mirrors the
  // backend's require_role so the UI disables what the API would refuse anyway -
  // the UI hint is a courtesy, the server check is the control.
  canAct: () => {
    const { user, loginRequired } = get();
    if (!loginRequired) return true;
    return ['operator', 'admin'].includes(user?.role);
  },
}));
