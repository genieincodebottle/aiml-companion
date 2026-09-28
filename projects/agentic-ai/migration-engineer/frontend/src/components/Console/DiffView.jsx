import React from 'react';
import { Box } from '@mui/material';
import { diffLines } from './format';

// Line tint by classified diff type. Additions green, deletions red, hunks/meta muted.
const LINE_SX = {
  add: { color: '#6ee7b7', bgcolor: 'rgba(52, 211, 153, 0.10)' },
  del: { color: '#fca5a5', bgcolor: 'rgba(248, 113, 113, 0.10)' },
  hunk: { color: '#7dd3fc', bgcolor: 'rgba(56, 189, 248, 0.08)' },
  meta: { color: '#94a3b8' },
  context: { color: 'text.secondary' },
};

// Renders a unified diff as a scrollable monospace block with per-line coloring.
// No syntax highlighter: lines are classified purely by their leading marker.
export default function DiffView({ diff, maxHeight = 320 }) {
  const lines = diffLines(diff);

  if (lines.length === 0) {
    return null;
  }

  return (
    <Box
      component="pre"
      sx={{
        m: 0,
        p: 1.5,
        maxHeight,
        overflow: 'auto',
        borderRadius: 1.5,
        bgcolor: '#0b0f17',
        border: '1px solid',
        borderColor: 'divider',
        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
        fontSize: 12,
        lineHeight: 1.6,
        whiteSpace: 'pre',
      }}
    >
      {lines.map((line, i) => (
        <Box
          key={i}
          component="span"
          sx={{
            display: 'block',
            px: 0.5,
            ...(LINE_SX[line.type] || LINE_SX.context),
          }}
        >
          {line.text || ' '}
        </Box>
      ))}
    </Box>
  );
}
