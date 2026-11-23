import { useEffect, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Alert,
  AppBar,
  Avatar,
  Box,
  Drawer,
  IconButton,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Typography,
  Divider,
  Chip,
  Menu,
  MenuItem,
  Tooltip,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Terminal as TerminalIcon,
  History as HistoryIcon,
  Bolt,
  Circle,
  AccountTree,
  Home as HomeIcon,
  LightMode,
  DarkMode,
  Logout,
} from '@mui/icons-material';
import { systemAPI } from '../../services/api';
import { useThemeMode } from '../../ThemeModeProvider';
import { useAuthStore } from '../../store/authStore';

const drawerWidth = 280;

const menuItems = [
  { text: 'Home', icon: <HomeIcon />, path: '/home', hint: 'What this is, and where to start' },
  {
    text: 'Console',
    icon: <TerminalIcon />,
    path: '/console',
    hint: 'Trigger a fault and watch it live',
  },
  {
    text: 'Workflow trace',
    icon: <AccountTree />,
    path: '/flow',
    hint: 'Replay any run stage by stage, or read how it all works',
  },
  {
    text: 'History',
    icon: <HistoryIcon />,
    path: '/history',
    hint: 'Past incidents and postmortems',
  },
];

export default function Layout({ children }) {
  const navigate = useNavigate();
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [health, setHealth] = useState({ status: null, model: null });
  const { mode, toggle } = useThemeMode();
  const { user, loginRequired, logout } = useAuthStore();
  const [userMenu, setUserMenu] = useState(null);

  useEffect(() => {
    let active = true;
    systemAPI
      .getHealth()
      .then((data) => {
        if (active) setHealth(data || {});
      })
      .catch(() => {
        if (active) setHealth({ status: 'down', model: null });
      });
    return () => {
      active = false;
    };
  }, []);

  const handleDrawerToggle = () => setMobileOpen(!mobileOpen);

  const online = health.status === 'ok';
  // "offline" here is the LLM mode (no Gemini key), not the network. The banner
  // below makes it impossible to mistake a rule-based run for model reasoning.
  const offlineMode = health.llm_mode === 'offline';

  const drawer = (
    <div>
      {/* Not a Toolbar and not noWrap: at a browser zoom above ~125% the product
          name was being clipped mid-word by the fixed toolbar height. */}
      <Box
        onClick={() => navigate('/home')}
        sx={{ px: 2, py: 2, cursor: 'pointer', display: 'flex', gap: 1, alignItems: 'flex-start' }}
      >
        <Bolt sx={{ color: 'primary.main', fontSize: 26, mt: 0.2, flexShrink: 0 }} />
        <Box sx={{ minWidth: 0 }}>
          <Typography
            variant="subtitle1"
            component="div"
            sx={{ fontWeight: 700, lineHeight: 1.25 }}
          >
            Incident Commander
          </Typography>
          <Typography variant="caption" color="text.secondary">
            Autonomous SRE Console
          </Typography>
        </Box>
      </Box>
      <Divider />

      <List sx={{ px: 1, py: 1 }}>
        {menuItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton
              selected={location.pathname === item.path}
              onClick={() => {
                navigate(item.path);
                setMobileOpen(false);
              }}
              sx={{
                borderRadius: 2,
                mb: 0.5,
                '&.Mui-selected': {
                  bgcolor: 'action.selected',
                  '&:hover': { bgcolor: 'action.selected' },
                },
              }}
            >
              <ListItemIcon sx={{ minWidth: 40 }}>{item.icon}</ListItemIcon>
              <ListItemText
                primary={item.text}
                secondary={item.hint}
                slotProps={{ secondary: { variant: 'caption', sx: { lineHeight: 1.3 } } }}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>

      <Box sx={{ position: 'absolute', bottom: 0, left: 0, right: 0, p: 2 }}>
        <Divider sx={{ mb: 2 }} />
        <Typography variant="caption" color="text.secondary" gutterBottom sx={{ display: 'block' }}>
          Backend
        </Typography>
        <Chip
          size="small"
          icon={<Circle sx={{ fontSize: 10 }} />}
          color={online ? 'success' : 'error'}
          label={online ? 'Connected' : 'Disconnected'}
          variant="outlined"
          sx={{ mb: 1 }}
        />
        {health.model && (
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
            Model: {health.model}
          </Typography>
        )}
      </Box>
    </div>
  );

  const titleForPath = () => {
    if (location.pathname.startsWith('/history')) return 'Incident History';
    if (location.pathname.startsWith('/flow')) return 'Workflow Trace';
    if (location.pathname.startsWith('/home')) return 'AI Incident Commander';
    return 'Live Incident Console';
  };

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <AppBar
        position="fixed"
        sx={{
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          ml: { sm: `${drawerWidth}px` },
          bgcolor: 'background.paper',
          // AppBar defaults its content to primary.contrastText (white). Against a
          // paper background that is invisible in light mode - it took the header
          // title and the theme toggle with it. The bar is a surface here, not a
          // primary-colored band, so its text follows the text palette.
          color: 'text.primary',
          borderBottom: '1px solid',
          borderColor: 'divider',
          boxShadow: 'none',
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { sm: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1, minWidth: 0, mr: 1 }}>
            {titleForPath()}
          </Typography>
          <Tooltip title={online ? 'Backend reachable' : 'Backend not reachable'}>
            <Chip
              size="small"
              icon={<Circle sx={{ fontSize: 10 }} />}
              color={online ? 'success' : 'error'}
              label={online ? 'Connected' : 'Disconnected'}
              variant="outlined"
            />
          </Tooltip>
          <Tooltip title={mode === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}>
            <IconButton
              onClick={toggle}
              color="inherit"
              sx={{ ml: 1 }}
              aria-label="Toggle color theme"
            >
              {mode === 'light' ? <DarkMode fontSize="small" /> : <LightMode fontSize="small" />}
            </IconButton>
          </Tooltip>

          {/* The account control only appears when the backend actually wants a
              login. With no database configured there are no users, and an avatar
              that opens a menu saying "anonymous" is pure noise. */}
          {loginRequired && (
            <>
              <Tooltip title={user ? `${user.username} (${user.role})` : 'Account'}>
                <IconButton onClick={(e) => setUserMenu(e.currentTarget)} sx={{ ml: 0.5 }}>
                  <Avatar sx={{ width: 30, height: 30, bgcolor: 'primary.main', fontSize: 14 }}>
                    {(user?.username || '?').slice(0, 1).toUpperCase()}
                  </Avatar>
                </IconButton>
              </Tooltip>
              <Menu
                anchorEl={userMenu}
                open={Boolean(userMenu)}
                onClose={() => setUserMenu(null)}
                anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
                transformOrigin={{ vertical: 'top', horizontal: 'right' }}
              >
                <Box sx={{ px: 2, py: 1.25, minWidth: 200 }}>
                  <Typography variant="subtitle2">{user?.username || 'Not signed in'}</Typography>
                  <Typography variant="caption" color="text.secondary">
                    {user?.role ? `role: ${user.role}` : ''}
                  </Typography>
                  {user?.role === 'viewer' && (
                    <Typography
                      variant="caption"
                      color="warning.main"
                      sx={{ display: 'block', mt: 0.5 }}
                    >
                      Viewers cannot approve or cancel.
                    </Typography>
                  )}
                </Box>
                <Divider />
                <MenuItem
                  onClick={async () => {
                    setUserMenu(null);
                    await logout();
                    navigate('/login', { replace: true });
                  }}
                >
                  <ListItemIcon sx={{ minWidth: 34 }}>
                    <Logout fontSize="small" />
                  </ListItemIcon>
                  Sign out
                </MenuItem>
              </Menu>
            </>
          )}
        </Toolbar>
      </AppBar>

      <Box component="nav" sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}>
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{ keepMounted: true }}
          sx={{
            display: { xs: 'block', sm: 'none' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
        >
          {drawer}
        </Drawer>
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', sm: 'block' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: drawerWidth,
              borderRight: '1px solid',
              borderColor: 'divider',
            },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>

      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: { xs: 2, md: 3 },
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          mt: 8,
          minHeight: '100vh',
          bgcolor: 'background.default',
        }}
      >
        {offlineMode && (
          <Alert severity="info" variant="outlined" sx={{ mb: 2 }}>
            <strong>Offline mode.</strong> No Gemini key is set, so the agents are a deterministic,
            rule-based runbook reading the same telemetry. The loop, guardrails and approval gate
            are all real. Add <code>GOOGLE_API_KEY</code> to <code>backend/.env</code> and restart
            to see real model reasoning.
          </Alert>
        )}
        {children}
      </Box>
    </Box>
  );
}
