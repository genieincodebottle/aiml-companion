// Shared visual metadata for repo statuses and activity entries.
// Keeps the fleet board and activity feeds color-consistent.

import {
  HourglassEmpty,
  Autorenew,
  RateReview,
  Gavel,
  MergeType,
  Cancel,
  ReportProblem,
  ErrorOutline,
  PlayArrow,
  Build,
  EditNote,
  FactCheck,
  Shield,
  CheckCircle,
} from '@mui/icons-material';

// Repo status -> label + icon. Color comes from repoStatusColor() in format.js.
export const REPO_STATUS_META = {
  queued: { label: 'Queued', icon: <HourglassEmpty fontSize="small" /> },
  migrating: { label: 'Migrating', icon: <Autorenew fontSize="small" /> },
  reviewing: { label: 'Reviewing', icon: <RateReview fontSize="small" /> },
  awaiting_approval: { label: 'Awaiting approval', icon: <Gavel fontSize="small" /> },
  pr_open: { label: 'PR open', icon: <MergeType fontSize="small" /> },
  rejected: { label: 'Rejected', icon: <Cancel fontSize="small" /> },
  escalated: { label: 'Escalated', icon: <ReportProblem fontSize="small" /> },
  failed: { label: 'Failed', icon: <ErrorOutline fontSize="small" /> },
};

export function repoStatusMeta(status) {
  return (
    REPO_STATUS_META[status] || {
      label: (status || 'unknown').replace(/_/g, ' '),
      icon: <HourglassEmpty fontSize="small" />,
    }
  );
}

// Activity kind -> icon (rendered in the per-repo live feed).
export const ACTIVITY_ICON = {
  step: <PlayArrow sx={{ fontSize: 15 }} />,
  tool: <Build sx={{ fontSize: 15 }} />,
  edit: <EditNote sx={{ fontSize: 15 }} />,
  tests: <FactCheck sx={{ fontSize: 15 }} />,
  review: <RateReview sx={{ fontSize: 15 }} />,
  guardrail: <Shield sx={{ fontSize: 15 }} />,
  await: <Gavel sx={{ fontSize: 15 }} />,
  pr: <MergeType sx={{ fontSize: 15 }} />,
  done: <CheckCircle sx={{ fontSize: 15 }} />,
};

export function activityIcon(kind) {
  return ACTIVITY_ICON[kind] || <PlayArrow sx={{ fontSize: 15 }} />;
}

// Pure helpers live in a dependency-free module; re-exported here so components
// can import visuals and helpers from one place.
export {
  repoStatusColor,
  jobStatusColor,
  modeColor,
  toneColor,
  diffLines,
  formatUsd,
  formatTime,
} from './format';
