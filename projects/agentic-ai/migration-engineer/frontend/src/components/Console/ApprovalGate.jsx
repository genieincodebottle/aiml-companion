import React, { useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Collapse,
  Divider,
  Stack,
  TextField,
  Typography,
} from '@mui/material';
import { Gavel, CheckCircle, Cancel, InsertDriveFileOutlined } from '@mui/icons-material';
import { useMigrationStore } from '../../store/migrationStore';
import DiffView from './DiffView';

// Human-in-the-loop gate for a single repo. Several of these can be on screen
// at once (fan-out concurrency), so each one owns its own note field and posts
// its own repo_id to /approve.
export default function ApprovalGate({ repoId, payload }) {
  const { deciding, decide } = useMigrationStore();
  const [showReject, setShowReject] = useState(false);
  const [note, setNote] = useState('');

  const busy = Boolean(deciding[repoId]);
  const files = payload.files_changed || [];

  const handleApprove = async () => {
    await decide(repoId, 'approve', note.trim() || undefined);
    setNote('');
    setShowReject(false);
  };

  const handleReject = async () => {
    if (!showReject) {
      setShowReject(true);
      return;
    }
    await decide(repoId, 'reject', note.trim() || undefined);
    setNote('');
    setShowReject(false);
  };

  return (
    <Card
      sx={{
        border: '1px solid',
        borderColor: 'warning.main',
        boxShadow: '0 0 0 3px rgba(251, 191, 36, 0.12)',
      }}
    >
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
          <Gavel color="warning" />
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            Approval required
          </Typography>
          <Chip
            size="small"
            label={payload.repo || repoId}
            sx={{ fontFamily: 'ui-monospace, monospace' }}
          />
        </Box>

        {payload.summary && (
          <Box
            sx={{
              p: 1.5,
              borderRadius: 1.5,
              bgcolor: 'rgba(148, 163, 184, 0.08)',
              mb: 1.5,
            }}
          >
            <Typography variant="caption" color="text.secondary" display="block">
              Agent summary
            </Typography>
            <Typography variant="body2">{payload.summary}</Typography>
          </Box>
        )}

        {files.length > 0 && (
          <Box sx={{ mb: 1.5 }}>
            <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.5 }}>
              Files changed ({files.length})
            </Typography>
            <Stack spacing={0.5}>
              {files.map((f) => (
                <Box key={f} sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}>
                  <InsertDriveFileOutlined sx={{ fontSize: 15, color: 'text.secondary' }} />
                  <Typography
                    variant="body2"
                    sx={{ fontFamily: 'ui-monospace, monospace', wordBreak: 'break-all' }}
                  >
                    {f}
                  </Typography>
                </Box>
              ))}
            </Stack>
          </Box>
        )}

        {payload.diff && (
          <Box sx={{ mb: 1.5 }}>
            <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.5 }}>
              Proposed diff
            </Typography>
            <DiffView diff={payload.diff} />
          </Box>
        )}

        <Collapse in={showReject}>
          <TextField
            fullWidth
            size="small"
            label="Rejection note (optional)"
            placeholder="Tell the agent why this change was rejected"
            value={note}
            onChange={(e) => setNote(e.target.value)}
            sx={{ mb: 1.5 }}
          />
        </Collapse>

        <Divider sx={{ mb: 1.5 }} />

        <Stack direction="row" spacing={1.5}>
          <Button
            fullWidth
            variant="contained"
            color="success"
            startIcon={busy ? <CircularProgress size={18} color="inherit" /> : <CheckCircle />}
            disabled={busy}
            onClick={handleApprove}
          >
            Approve
          </Button>
          <Button
            fullWidth
            variant="outlined"
            color="error"
            startIcon={<Cancel />}
            disabled={busy}
            onClick={handleReject}
          >
            {showReject ? 'Confirm reject' : 'Reject'}
          </Button>
        </Stack>

        <Collapse in={busy}>
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            Submitting decision to the migration run...
          </Typography>
        </Collapse>
      </CardContent>
    </Card>
  );
}
