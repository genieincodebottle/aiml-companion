import React from 'react';
import { Box, Card, CardContent, Chip, Divider, Stack, Typography } from '@mui/material';
import { TaskAlt, Token, Paid } from '@mui/icons-material';
import { repoStatusColor, modeColor, formatUsd } from './format';
import { repoStatusMeta } from './repoMeta';

function Stat({ icon, label, value, color = 'text.primary' }) {
  return (
    <Box sx={{ minWidth: 96 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, color: 'text.secondary' }}>
        {icon}
        <Typography variant="caption">{label}</Typography>
      </Box>
      <Typography variant="h6" sx={{ color, mt: 0.25 }}>
        {value}
      </Typography>
    </Box>
  );
}

// Results banner rendered once a job_summary event arrives (or in history).
export default function JobSummaryBanner({ summary }) {
  if (!summary) return null;

  const byStatus = summary.by_status || {};
  const total = summary.repos_total ?? 0;
  const merged = summary.repos_merged ?? 0;

  return (
    <Card sx={{ borderTop: '3px solid', borderColor: 'success.main' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5, flexWrap: 'wrap' }}>
          <TaskAlt color="success" />
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            Migration complete
          </Typography>
          {summary.mode && (
            <Chip size="small" color={modeColor(summary.mode)} label={summary.mode} variant="outlined" />
          )}
        </Box>

        <Stack direction="row" spacing={3} flexWrap="wrap" useFlexGap sx={{ mb: 2 }}>
          <Stat
            icon={<TaskAlt sx={{ fontSize: 16 }} />}
            label="Merged"
            value={`${merged} / ${total}`}
            color="success.light"
          />
          <Stat
            icon={<Token sx={{ fontSize: 16 }} />}
            label="Tokens"
            value={Number(summary.total_tokens || 0).toLocaleString()}
          />
          <Stat
            icon={<Paid sx={{ fontSize: 16 }} />}
            label="Total cost"
            value={formatUsd(summary.total_cost_usd)}
            color="success.light"
          />
        </Stack>

        <Divider sx={{ mb: 1.5 }} />

        <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
          Outcome by status
        </Typography>
        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
          {Object.keys(byStatus).length === 0 ? (
            <Typography variant="body2" color="text.secondary">
              No repositories reported.
            </Typography>
          ) : (
            Object.entries(byStatus).map(([status, count]) => (
              <Chip
                key={status}
                size="small"
                color={repoStatusColor(status)}
                icon={repoStatusMeta(status).icon}
                label={`${repoStatusMeta(status).label}: ${count}`}
              />
            ))
          )}
        </Stack>
      </CardContent>
    </Card>
  );
}
