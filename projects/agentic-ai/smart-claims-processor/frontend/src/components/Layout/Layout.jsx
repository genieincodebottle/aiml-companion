import React from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import {
  Box, Drawer, List, ListItemButton, ListItemIcon, ListItemText, Typography,
  AppBar, Toolbar, IconButton, Avatar, Chip, Badge, Divider, Menu, MenuItem, Tooltip,
} from '@mui/material';
import {
  Dashboard, Description, Gavel, Security, Assessment, People, Settings, Logout,
  AccountBalance, ReceiptLong, HowToVote, Analytics, Menu as MenuIcon,
} from '@mui/icons-material';
import useAuthStore from '../../store/authStore';
import useClaimsStore from '../../store/claimsStore';

const DRAWER_WIDTH = 260;

// Backend roles: user | reviewer | admin
const ROLE_COLORS = {
  user: '#3B82F6',      // blue - claimant
  reviewer: '#F59E0B',  // amber - HITL approver
  admin: '#EF4444',     // red - admin
};

const ROLE_LABELS = {
  user: 'Claimant',
  reviewer: 'Reviewer',
  admin: 'Admin',
};

export default function Layout() {
  const navigate = useNavigate();
  const location = useLocation();
  const {
    user, logout, isClaimant, canReviewClaims, canReviewHITL,
    canReviewAppeals, canViewAnalytics, canManageUsers,
  } = useAuthStore();
  const hitlStats = useClaimsStore((s) => s.hitlStats);
  const [mobileOpen, setMobileOpen] = React.useState(false);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const claimantOnly = isClaimant();       // role === 'user'
  const reviewerOrAdmin = canReviewClaims(); // role >= reviewer
  const adminOnly = canManageUsers();       // role === 'admin'

  const navItems = [
    // Claimant items (hidden from reviewers/admins - they use the staff views)
    { path: '/my-claims',    label: 'My Claims',        icon: <Description />, show: claimantOnly },
    { path: '/submit-claim', label: 'Submit Claim',     icon: <ReceiptLong />, show: claimantOnly },
    { path: '/my-appeals',   label: 'My Appeals',       icon: <HowToVote />,   show: claimantOnly },
    { divider: true, show: reviewerOrAdmin },

    // Reviewer + admin items
    { path: '/dashboard',  label: 'Claims Dashboard',  icon: <Dashboard />, show: reviewerOrAdmin },
    { path: '/hitl-queue', label: 'HITL Queue', icon: <Security />,  show: canReviewHITL(),   badge: hitlStats?.pending_total, tooltip: 'HITL (Human-In-The-Loop) Review Queue' },
    { path: '/appeals',    label: 'Appeals Review',    icon: <Gavel />,     show: canReviewAppeals() },
    { path: '/analytics',  label: 'Analytics',         icon: <Analytics />, show: canViewAnalytics() },

    // Admin-only
    { divider: true, show: adminOnly },
    { path: '/user-management', label: 'User Management', icon: <People />, show: adminOnly },
    { path: '/policy-management', label: 'Policy Management', icon: <Description />, show: adminOnly },

    // Everyone
    { divider: true, show: true },
    { path: '/settings', label: 'Settings', icon: <Settings />, show: true },
  ];

  const roleColor = ROLE_COLORS[user?.role] || '#64748B';

  const drawer = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', bgcolor: '#0F172A' }}>
      {/* Brand */}
      <Box sx={{ p: 2.5, display: 'flex', alignItems: 'center', gap: 1.5 }}>
        <AccountBalance sx={{ fontSize: 32, color: '#3B82F6' }} />
        <Box>
          <Typography variant="subtitle1" fontWeight={700} color="#F1F5F9" lineHeight={1.2}>Smart Claims</Typography>
          <Typography variant="caption" color="#64748B">Insurance AI</Typography>
        </Box>
      </Box>
      <Divider sx={{ borderColor: '#1E293B' }} />

      {/* User info */}
      <Box sx={{ p: 2, display: 'flex', alignItems: 'center', gap: 1.5 }}>
        <Avatar sx={{ width: 36, height: 36, bgcolor: roleColor, fontSize: '0.9rem', fontWeight: 600 }}>
          {user?.username?.[0]?.toUpperCase() || '?'}
        </Avatar>
        <Box sx={{ flex: 1, minWidth: 0 }}>
          <Typography variant="body2" fontWeight={600} color="#F1F5F9" noWrap>{user?.username}</Typography>
          <Chip label={ROLE_LABELS[user?.role] || user?.role} size="small"
            sx={{ height: 20, fontSize: '0.7rem', fontWeight: 600, bgcolor: `${roleColor}20`, color: roleColor, border: `1px solid ${roleColor}40` }}
          />
        </Box>
      </Box>
      <Divider sx={{ borderColor: '#1E293B' }} />

      {/* Nav items */}
      <List sx={{ flex: 1, px: 1, py: 1 }}>
        {navItems.filter((n) => n.show).map((item, i) =>
          item.divider ? (
            <Divider key={i} sx={{ my: 1, borderColor: '#1E293B' }} />
          ) : (
            <Tooltip title={item.tooltip || ''} placement="right" arrow>
              <ListItemButton
                key={item.path}
                selected={location.pathname === item.path}
                onClick={() => { navigate(item.path); setMobileOpen(false); }}
                sx={{
                  borderRadius: 2, mb: 0.5, py: 1,
                  '&.Mui-selected': { bgcolor: '#1E293B', '& .MuiListItemIcon-root': { color: '#3B82F6' }, '& .MuiListItemText-primary': { color: '#F1F5F9', fontWeight: 600 } },
                  '&:hover': { bgcolor: '#1E293B80' },
                }}
              >
                <ListItemIcon sx={{ color: '#64748B', minWidth: 40 }}>
                  {item.badge ? <Badge badgeContent={item.badge} color="error">{item.icon}</Badge> : item.icon}
                </ListItemIcon>
                <ListItemText primary={item.label} sx={{ '& .MuiListItemText-primary': { fontSize: '0.875rem', color: '#94A3B8' } }} />
              </ListItemButton>
            </Tooltip>
          )
        )}
      </List>

      {/* Logout */}
      <Divider sx={{ borderColor: '#1E293B' }} />
      <ListItemButton onClick={handleLogout} sx={{ m: 1, borderRadius: 2, py: 1 }}>
        <ListItemIcon sx={{ color: '#EF4444', minWidth: 40 }}><Logout /></ListItemIcon>
        <ListItemText primary="Sign Out" sx={{ '& .MuiListItemText-primary': { color: '#EF4444', fontSize: '0.875rem', fontWeight: 500 } }} />
      </ListItemButton>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: '#0F172A' }}>
      {/* Mobile app bar */}
      <AppBar position="fixed" sx={{ display: { md: 'none' }, bgcolor: '#0F172A', borderBottom: '1px solid #1E293B' }}>
        <Toolbar>
          <IconButton onClick={() => setMobileOpen(!mobileOpen)} sx={{ color: '#F1F5F9' }}><MenuIcon /></IconButton>
          <Typography variant="subtitle1" fontWeight={600} color="#F1F5F9" sx={{ ml: 1 }}>Smart Claims</Typography>
        </Toolbar>
      </AppBar>

      {/* Drawer */}
      <Drawer variant="temporary" open={mobileOpen} onClose={() => setMobileOpen(false)}
        sx={{ display: { xs: 'block', md: 'none' }, '& .MuiDrawer-paper': { width: DRAWER_WIDTH, bgcolor: '#0F172A', borderRight: '1px solid #1E293B' } }}
      >
        {drawer}
      </Drawer>
      <Drawer variant="permanent"
        sx={{ display: { xs: 'none', md: 'block' }, '& .MuiDrawer-paper': { width: DRAWER_WIDTH, bgcolor: '#0F172A', borderRight: '1px solid #1E293B' } }}
      >
        {drawer}
      </Drawer>

      {/* Main content */}
      <Box component="main" sx={{ flex: 1, ml: { md: `${DRAWER_WIDTH}px` }, mt: { xs: '56px', md: 0 }, p: 3, minHeight: '100vh' }}>
        <Outlet />
      </Box>
    </Box>
  );
}
