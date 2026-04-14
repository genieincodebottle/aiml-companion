import { create } from 'zustand';

// Must match api/security.py SEED_USERS roles.
// Hierarchy: user < reviewer < admin
const ROLES = {
  user: 'user',
  reviewer: 'reviewer',
  admin: 'admin',
};

const ROLE_HIERARCHY = [ROLES.user, ROLES.reviewer, ROLES.admin];

const useAuthStore = create((set, get) => ({
  user: JSON.parse(localStorage.getItem('user') || 'null'),
  token: localStorage.getItem('token') || null,
  isAuthenticated: !!localStorage.getItem('token'),

  login: (user, token) => {
    localStorage.setItem('user', JSON.stringify(user));
    localStorage.setItem('token', token);
    set({ user, token, isAuthenticated: true });
  },

  logout: () => {
    localStorage.removeItem('user');
    localStorage.removeItem('token');
    set({ user: null, token: null, isAuthenticated: false });
  },

  updateUser: (user) => {
    localStorage.setItem('user', JSON.stringify(user));
    set({ user });
  },

  // Role checks
  hasRole: (minRole) => {
    const user = get().user;
    if (!user) return false;
    const userIdx = ROLE_HIERARCHY.indexOf(user.role);
    const minIdx = ROLE_HIERARCHY.indexOf(minRole);
    if (userIdx < 0 || minIdx < 0) return false;
    return userIdx >= minIdx;
  },

  isClaimant: () => get().user?.role === ROLES.user,
  isReviewer: () => get().user?.role === ROLES.reviewer,
  isAdmin: () => get().user?.role === ROLES.admin,

  // Permission checks - map to the backend's role gates
  canSubmitClaims: () => get().user?.role === ROLES.user,     // claimants file claims
  canReviewClaims: () => get().hasRole(ROLES.reviewer),        // reviewers + admins see the claims dashboard
  canReviewHITL: () => get().hasRole(ROLES.reviewer),          // reviewers + admins approve HITL tickets
  canReviewAppeals: () => get().hasRole(ROLES.reviewer),       // reviewers + admins review appeals
  canViewAnalytics: () => get().hasRole(ROLES.reviewer),       // reviewers + admins see analytics
  canManageUsers: () => get().user?.role === ROLES.admin,      // admin-only
  canChangeLLMProvider: () => get().user?.role === ROLES.admin, // admin-only
}));

export { ROLES, ROLE_HIERARCHY };
export default useAuthStore;
