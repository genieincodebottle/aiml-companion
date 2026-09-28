import React, { useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Collapse,
  Divider,
  Stack,
  Tooltip,
  Typography,
} from '@mui/material';
import {
  Autorenew,
  Build,
  DescriptionOutlined,
  CheckCircle,
  Cancel,
  Shield,
  UnfoldMore,
  UnfoldLess,
} from '@mui/icons-material';
import RepoActivity from './RepoActivity';
import DiffView from './DiffView';
import { repoStatusMeta } from './repoMeta';
import { repoStatusColor, formatUsd } from './format';

function Metric({ icon, label, value }) {
  return (
    <Box sx={{ minWidth: 64 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, color: 'text.secondary' }}>
        {icon}
        <Typography variant="caption">{label}</Typography>
      </Box>
      <Typography variant="subtitle2" sx={{ mt: 0.25 }}>
        {value}
      </Typography>
    </Box>
  );
}

// One lane per repository. Shows live status, counters, tests, guardrail blocks,
// and a scrollable activity feed. In read-only mode it also exposes the diff.
export default function RepoCard({ repo, readOnly = false }) {
  const [showDiff, setShowDiff] = useState(false);
  const meta = repoStatusMeta(repo.status);
  const color = repoStatusColor(repo.status);

  const testsChip =
    repo.tests_passing == null ? null : (
      <Chip
        size="small"
        variant="outlined"
        color={repo.tests_passing ? 'success' : 'error'}
        icon={repo.tests_passing ? <CheckCircle sx={{ fontSize: 14 }} /> : <Cancel sx={{ fontSize: 14 }} />}
        label={repo.tests_passing ? 'tests pass' : 'tests fail'}
      />
    );

  return (
    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardContent sx={{ pb: 1.5, flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
          <Typography
            variant="subtitle1"
            sx={{ flexGrow: 1, fontFamily: 'ui-monospace, monospace', fontWeight: 700, wordBreak: 'break-all' }}
          >
            {repo.name}
          </Typography>
          <Chip size="small" color={color} icon={meta.icon} label={meta.label} />
        </Box>

        <Stack direction="row" spacing={2} sx={{ mb: 1.5 }} flexWrap="wrap" useFlexGap>
          <Metric icon={<Autorenew sx={{ fontSize: 15 }} />} label="Steps" value={repo.steps} />
          <Metric icon={<Build sx={{ fontSize: 15 }} />} label="Tools" value={repo.tool_calls} />
          <Metric
            icon={<DescriptionOutlined sx={{ fontSize: 15 }} />}
            label="Files"
            value={repo.files_changed.length}
          />
          <Metric icon={<Build sx={{ fontSize: 15 }} />} label="Cost" value={formatUsd(repo.cost_usd)} />
        </Stack>

        <Stack direction="row" spacing={1} sx={{ mb: 1.5 }} flexWrap="wrap" useFlexGap>
          {testsChip}
          {repo.review_verdict && (
            <Chip
              size="small"
              variant="outlined"
              color={repo.review_verdict.approve ? 'success' : 'error'}
              label={`review ${repo.review_verdict.approve ? 'approved' : 'rejected'}`}
            />
          )}
          {repo.review_verdict?.tampered_with_tests && (
            <Chip size="small" color="error" icon={<Shield sx={{ fontSize: 14 }} />} label="tests tampered" />
          )}
          {repo.pr && (
            <Tooltip title={repo.pr.url || repo.pr.title || ''}>
              <Chip
                size="small"
                color="success"
                variant="outlined"
                label={`PR #${repo.pr.number}`}
                {...(repo.pr.url && repo.pr.url.startsWith('http')
                  ? { component: 'a', href: repo.pr.url, target: '_blank', rel: 'noreferrer', clickable: true }
                  : {})}
              />
            </Tooltip>
          )}
        </Stack>

        {repo.guardrail_blocks.length > 0 && (
          <Box
            sx={{
              p: 1,
              mb: 1.5,
              borderRadius: 1.5,
              border: '1px solid',
              borderColor: 'warning.main',
              bgcolor: 'rgba(251, 191, 36, 0.10)',
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, mb: 0.5 }}>
              <Shield sx={{ fontSize: 16, color: 'warning.main' }} />
              <Typography variant="caption" fontWeight={700} color="warning.main">
                Guardrail blocked {repo.guardrail_blocks.length}{' '}
                {repo.guardrail_blocks.length === 1 ? 'action' : 'actions'}
              </Typography>
            </Box>
            {repo.guardrail_blocks.map((g, i) => (
              <Typography key={i} variant="caption" display="block" color="text.secondary">
                {g.tool}
                {g.target ? ` on ${g.target}` : ''}: {g.reason}
              </Typography>
            ))}
          </Box>
        )}

        {repo.escalation_reason && (
          <Typography variant="caption" color="error.light" display="block" sx={{ mb: 1.5 }}>
            Escalation: {repo.escalation_reason}
          </Typography>
        )}

        <Divider sx={{ mb: 1 }} />
        <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.75 }}>
          Activity
        </Typography>
        <RepoActivity activity={repo.activity} autoScroll={!readOnly} />

        {readOnly && repo.diff && (
          <Box sx={{ mt: 1.5 }}>
            <Button
              size="small"
              startIcon={showDiff ? <UnfoldLess /> : <UnfoldMore />}
              onClick={() => setShowDiff((v) => !v)}
            >
              {showDiff ? 'Hide diff' : 'View diff'}
            </Button>
            <Collapse in={showDiff}>
              <Box sx={{ mt: 1 }}>
                <DiffView diff={repo.diff} maxHeight={260} />
              </Box>
            </Collapse>
          </Box>
        )}
      </CardContent>
    </Card>
  );
}
