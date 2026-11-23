import { createTheme, darken } from '@mui/material/styles';

// Light and dark palettes for the ops console.
//
// Light is the default. The original console was dark-only, which suits a wall
// display but not the two places this actually gets read: a laptop in a bright
// room, and a projector in a lit lecture theatre. Dark stays a first-class option
// rather than an afterthought, and the choice persists.
//
// Palette rule (both modes): blue/teal for neutral chrome, amber for "a human
// should look", green for good, red for bad, slate for bookkeeping. No violet or
// magenta - a color in this console is supposed to mean something.

const SHARED = {
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h4: { fontWeight: 600 },
    h5: { fontWeight: 600 },
    h6: { fontWeight: 600 },
  },
  shape: { borderRadius: 10 },
};

const DARK = {
  primary: { main: '#4f9eff', light: '#7db6ff', dark: '#2d7be0' },
  secondary: { main: '#22d3ee', light: '#67e8f9', dark: '#0891b2' },
  success: { main: '#34d399', light: '#6ee7b7', dark: '#10b981' },
  warning: { main: '#fbbf24', light: '#fcd34d', dark: '#d97706' },
  error: { main: '#f87171', light: '#fca5a5', dark: '#ef4444' },
  info: { main: '#38bdf8', light: '#7dd3fc', dark: '#0ea5e9' },
  background: { default: '#0b0f17', paper: '#141b29' },
  divider: 'rgba(148, 163, 184, 0.16)',
};

// Light-mode hues are pulled down a few steps. The dark palette's colors are
// tuned for glowing on near-black and become illegible pastels on white, so this
// is a genuinely different set rather than the same hex values re-used.
const LIGHT = {
  primary: { main: '#1d6fd4', light: '#4f9eff', dark: '#14539f' },
  secondary: { main: '#0e8ca8', light: '#22d3ee', dark: '#0b6b80' },
  success: { main: '#0f9d6e', light: '#34d399', dark: '#0a7550' },
  warning: { main: '#b45309', light: '#f59e0b', dark: '#92400e' },
  error: { main: '#dc2626', light: '#f87171', dark: '#b91c1c' },
  info: { main: '#0284c7', light: '#38bdf8', dark: '#0369a1' },
  background: { default: '#f5f7fa', paper: '#ffffff' },
  divider: 'rgba(71, 85, 105, 0.18)',
  text: { primary: '#0f172a', secondary: '#475569' },
};

export function createAppTheme(mode) {
  const isLight = mode === 'light';
  return createTheme({
    ...SHARED,
    palette: { mode, ...(isLight ? LIGHT : DARK) },
    components: {
      MuiCard: {
        styleOverrides: {
          root: {
            borderRadius: 12,
            backgroundImage: 'none',
            border: '1px solid',
            borderColor: isLight ? 'rgba(71, 85, 105, 0.14)' : 'rgba(148, 163, 184, 0.14)',
            ...(isLight ? { boxShadow: '0 1px 2px rgba(15, 23, 42, 0.06)' } : {}),
          },
        },
      },
      MuiButton: {
        styleOverrides: {
          root: { textTransform: 'none', borderRadius: 8, fontWeight: 600 },
        },
      },
      // MUI uppercases tab labels by default, which turns a written sentence into
      // SHOUTING and costs the word shapes people actually read by. Labels are
      // authored in sentence case; render them that way.
      MuiTab: {
        styleOverrides: {
          root: { textTransform: 'none', fontWeight: 600 },
        },
      },
      MuiAppBar: { styleOverrides: { root: { backgroundImage: 'none' } } },
    },
  });
}

// Gradients.
//
// One brand gradient, reused for the login panel, the primary CTA and the home
// hero, with a darker twin for :hover. Keeping it to a single pair is what stops
// a gradient reading as decoration - it becomes the product's signature instead of
// three unrelated colour washes.
export const GRADIENTS = {
  brand: 'linear-gradient(135deg, #1d6fd4 0%, #2f8fd8 55%, #22a5c4 100%)',
  brandHover: 'linear-gradient(135deg, #14539f 0%, #1d6fd4 55%, #0e8ca8 100%)',
  // The full-page wash behind the login card. Light and dark differ in more than
  // brightness: on dark the same pale stops would glow like a lightbox.
  page: (mode) =>
    mode === 'light'
      ? 'linear-gradient(135deg, #e8f0fb 0%, #eef6fa 45%, #f5f7fa 100%)'
      : 'linear-gradient(135deg, #0b1220 0%, #101a2b 45%, #0b0f17 100%)',
  // A soft tint for hero surfaces that sit on the normal page background.
  hero: (mode) =>
    mode === 'light'
      ? 'linear-gradient(135deg, rgba(29,111,212,0.10) 0%, rgba(34,165,196,0.07) 50%, transparent 100%)'
      : 'linear-gradient(135deg, rgba(79,158,255,0.14) 0%, rgba(34,211,238,0.08) 50%, transparent 100%)',
};

// The agent and event accent colors live outside the MUI palette (see
// agentMeta.jsx) because they identify *who acted*, not a semantic role. They are
// authored for dark backgrounds, so on light they get darkened to stay readable
// against white rather than being maintained as a duplicate table that drifts.
export function toneFor(color, mode) {
  if (!color || mode !== 'light') return color;
  return darken(color, 0.3);
}

// Translucent fill derived from an accent, for chips and icon bubbles.
export function tintFor(color, mode) {
  return `${toneFor(color, mode)}${mode === 'light' ? '1f' : '22'}`;
}
