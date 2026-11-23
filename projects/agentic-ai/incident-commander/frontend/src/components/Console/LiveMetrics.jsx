import { useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Chip,
  Divider,
  LinearProgress,
  Stack,
  Typography,
} from '@mui/material';
import { Insights, Loop, Token, Paid } from '@mui/icons-material';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts';
import { useTheme } from '@mui/material/styles';
import { useIncidentStore } from '../../store/incidentStore';
import { statusColor } from './agentMeta';

function Stat({ icon, label, value, color = 'text.primary' }) {
  return (
    <Box sx={{ flex: 1, minWidth: 96 }}>
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

export default function LiveMetrics() {
  const { status, iteration, maxIterations, totalTokens, totalCostUsd, events } =
    useIncidentStore();
  const theme = useTheme();
  const axis = theme.palette.text.secondary;

  // Build a cumulative cost-by-iteration series from token_usage events. Each
  // payload already carries the incident's CUMULATIVE totals (same contract the
  // store relies on), so the latest event for an iteration IS that iteration's
  // point - it is assigned, not summed. Summing here multiplied the plotted cost
  // by the number of emits, so the chart disagreed with the Cost stat beside it.
  const costSeries = useMemo(() => {
    const byIter = new Map();
    events
      .filter((e) => e.type === 'token_usage')
      .forEach((e) => {
        const iter = e.payload?.iteration ?? e.iteration ?? 0;
        byIter.set(iter, {
          iteration: iter,
          cost: Number(Number(e.payload?.cost_usd || 0).toFixed(4)),
          tokens: Number(e.payload?.tokens || 0),
        });
      });
    return Array.from(byIter.values()).sort((a, b) => a.iteration - b.iteration);
  }, [events]);

  const progress =
    maxIterations && maxIterations > 0 ? Math.min(100, (iteration / maxIterations) * 100) : null;

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
          <Insights color="info" />
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            Live Metrics
          </Typography>
          <Chip
            size="small"
            color={status ? statusColor(status) : 'default'}
            label={status ? status.replace(/_/g, ' ') : 'idle'}
            variant={status ? 'filled' : 'outlined'}
          />
        </Box>

        <Stack direction="row" spacing={2} sx={{ flexWrap: 'wrap', mb: 2 }} useFlexGap>
          <Stat
            icon={<Loop sx={{ fontSize: 16 }} />}
            label="Iterations"
            value={maxIterations ? `${iteration} / ${maxIterations}` : iteration}
          />
          <Stat
            icon={<Token sx={{ fontSize: 16 }} />}
            label="Tokens"
            value={totalTokens.toLocaleString()}
          />
          <Stat
            icon={<Paid sx={{ fontSize: 16 }} />}
            label="Cost"
            value={`$${Number(totalCostUsd).toFixed(4)}`}
            color="success.light"
          />
        </Stack>

        {progress != null && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="caption" color="text.secondary">
              Loop budget
            </Typography>
            <LinearProgress
              variant="determinate"
              value={progress}
              sx={{ height: 8, borderRadius: 999, mt: 0.5 }}
            />
          </Box>
        )}

        <Divider sx={{ mb: 1.5 }} />

        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
          Cumulative cost per iteration
        </Typography>
        {costSeries.length === 0 ? (
          <Box
            sx={{
              height: 140,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'text.secondary',
            }}
          >
            <Typography variant="body2">No token usage reported yet</Typography>
          </Box>
        ) : (
          <Box sx={{ height: 140 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={costSeries} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
                <defs>
                  <linearGradient id="costFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#34d399" stopOpacity={0.5} />
                    <stop offset="100%" stopColor="#34d399" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.15)" />
                <XAxis dataKey="iteration" stroke={axis} fontSize={11} tickLine={false} />
                <YAxis stroke={axis} fontSize={11} tickLine={false} width={48} />
                <Tooltip
                  // recharts renders outside the MUI tree, so its tooltip does
                  // not inherit the palette. Hardcoded dark here meant dark text
                  // on a dark chip once light mode existed.
                  contentStyle={{
                    background: theme.palette.background.paper,
                    color: theme.palette.text.primary,
                    border: `1px solid ${theme.palette.divider}`,
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  formatter={(value, name) =>
                    name === 'cost' ? [`$${value}`, 'cost'] : [value, name]
                  }
                />
                <Area
                  type="monotone"
                  dataKey="cost"
                  stroke="#34d399"
                  strokeWidth={2}
                  fill="url(#costFill)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </Box>
        )}
      </CardContent>
    </Card>
  );
}
