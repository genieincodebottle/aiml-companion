import React, { useEffect, useRef } from 'react';
import { Box, Typography } from '@mui/material';
import { activityIcon } from './repoMeta';
import { toneColor, formatTime } from './format';

// Compact, scrollable feed of one repo's events. Guardrail blocks and failed
// tests / rejected reviews are tinted with their tone so they stand out.
export default function RepoActivity({ activity = [], autoScroll = true, maxHeight = 200 }) {
  const scrollRef = useRef(null);

  useEffect(() => {
    if (!autoScroll) return;
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [activity.length, autoScroll]);

  if (activity.length === 0) {
    return (
      <Typography variant="caption" color="text.secondary">
        No activity yet.
      </Typography>
    );
  }

  return (
    <Box
      ref={scrollRef}
      sx={{
        maxHeight,
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: 0.75,
      }}
    >
      {activity.map((item, i) => {
        const prominent = item.kind === 'guardrail';
        const color = `${toneColor(item.tone)}.main`;
        return (
          <Box
            key={i}
            sx={{
              display: 'flex',
              alignItems: 'flex-start',
              gap: 0.75,
              px: prominent ? 1 : 0.5,
              py: prominent ? 0.75 : 0.25,
              borderRadius: 1,
              ...(prominent && {
                border: '1px solid',
                borderColor: 'warning.main',
                bgcolor: 'rgba(251, 191, 36, 0.10)',
              }),
            }}
          >
            <Box sx={{ color, mt: '1px', flexShrink: 0 }}>{activityIcon(item.kind)}</Box>
            <Box sx={{ flexGrow: 1, minWidth: 0 }}>
              <Typography
                variant="caption"
                sx={{
                  color: item.tone && item.tone !== 'info' ? color : 'text.primary',
                  fontWeight: prominent ? 700 : 500,
                  wordBreak: 'break-word',
                }}
              >
                {item.agent ? `${item.agent}: ` : ''}
                {item.text}
              </Typography>
            </Box>
            {item.ts && (
              <Typography variant="caption" color="text.secondary" sx={{ flexShrink: 0 }}>
                {formatTime(item.ts)}
              </Typography>
            )}
          </Box>
        );
      })}
    </Box>
  );
}
