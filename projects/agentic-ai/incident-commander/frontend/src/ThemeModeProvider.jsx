import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { createAppTheme } from './theme';

const STORAGE_KEY = 'ic.themeMode';

const ThemeModeContext = createContext({ mode: 'light', setMode: () => {}, toggle: () => {} });

// Reads the saved choice, else the OS preference, else light.
//
// The OS fallback is only consulted when the user has never chosen here: once
// someone picks a mode in this app it wins, because a console that flips itself
// when the laptop crosses sunset is worse than one that is occasionally wrong.
function initialMode() {
  try {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (saved === 'light' || saved === 'dark') return saved;
    if (window.matchMedia?.('(prefers-color-scheme: dark)').matches) return 'dark';
  } catch {
    // Private mode or a blocked storage API: fall through to the default.
  }
  return 'light';
}

export function ThemeModeProvider({ children }) {
  const [mode, setMode] = useState(initialMode);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, mode);
    } catch {
      // Non-fatal: the theme still applies for this session.
    }
  }, [mode]);

  const toggle = useCallback(() => setMode((m) => (m === 'light' ? 'dark' : 'light')), []);
  const theme = useMemo(() => createAppTheme(mode), [mode]);
  const value = useMemo(() => ({ mode, setMode, toggle }), [mode, toggle]);

  return (
    <ThemeModeContext.Provider value={value}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        {children}
      </ThemeProvider>
    </ThemeModeContext.Provider>
  );
}

export function useThemeMode() {
  return useContext(ThemeModeContext);
}
