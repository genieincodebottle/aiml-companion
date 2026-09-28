import React, { useEffect, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  AppBar,
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
  Tooltip,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Terminal as TerminalIcon,
  History as HistoryIcon,
  Hub,
  Circle,
} from '@mui/icons-material';
import { useMigrationStore } from '../../store/migrationStore';
import { modeColor } from '../Console/format';

const drawerWidth = 248;

const menuItems = [
  { text: 'Console', icon: <TerminalIcon />, path: '/console' },
  { text: 'History', icon: <HistoryIcon />, path: '/history' },
];

export default function Layout({ children }) {
  const navigate = useNavigate();
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);
  const { health, mode, loadHealth } = useMigrationStore();

  useEffect(() => {
    loadHealth();
  }, [loadHealth]);

  const handleDrawerToggle = () => setMobileOpen(!mobileOpen);

  const online = health?.status === 'ok';
  const modeText = mode || (online ? 'unknown' : 'offline');

  const drawer = (
    <div>
      <Toolbar sx={{ flexDirection: 'column', alignItems: 'flex-start', py: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
          <Hub sx={{ mr: 1, color: 'primary.main', fontSize: 28 }} />
          <Typography variant="h6" noWrap component="div">
            Migration Engineer
          </Typography>
        </Box>
        <Typography variant="caption" color="text.secondary">
          Autonomous fleet migration console
        </Typography>
      </Toolbar>
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
                  bgcolor: 'rgba(79, 158, 255, 0.16)',
                  '&:hover': { bgcolor: 'rgba(79, 158, 255, 0.24)' },
                },
              }}
            >
              <ListItemIcon sx={{ minWidth: 40 }}>{item.icon}</ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>

      <Box sx={{ position: 'absolute', bottom: 0, left: 0, right: 0, p: 2 }}>
        <Divider sx={{ mb: 2 }} />
        <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
          Execution mode
        </Typography>
        <Chip
          size="small"
          icon={<Circle sx={{ fontSize: 10 }} />}
          color={modeColor(mode)}
          label={modeText}
          variant="outlined"
          sx={{ mb: 1 }}
        />
        {health?.model && (
          <Typography variant="caption" color="text.secondary" display="block">
            Model: {health.model}
          </Typography>
        )}
        {health?.detail && (
          <Typography variant="caption" color="text.secondary" display="block">
            {health.detail}
          </Typography>
        )}
      </Box>
    </div>
  );

  const titleForPath = () => {
    if (location.pathname.startsWith('/history')) return 'Migration History';
    return 'Live Migration Console';
  };

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <AppBar
        position="fixed"
        sx={{
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          ml: { sm: `${drawerWidth}px` },
          bgcolor: 'background.paper',
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
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            Autonomous Migration Engineer
          </Typography>
          <Tooltip title={titleForPath()}>
            <Chip
              size="small"
              icon={<Circle sx={{ fontSize: 10 }} />}
              color={modeColor(mode)}
              label={modeText}
              variant="outlined"
            />
          </Tooltip>
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
        {children}
      </Box>
    </Box>
  );
}
