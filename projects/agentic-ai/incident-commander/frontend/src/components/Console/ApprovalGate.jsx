import { useState } from 'react';
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
import { Gavel, CheckCircle, Cancel, ShieldOutlined } from '@mui/icons-material';
import { useIncidentStore } from '../../store/incidentStore';
import { riskColor } from './agentMeta';

// Prominent human-in-the-loop gate. Shows the pending risky action and
// posts the operator decision to /approve.
export default function ApprovalGate() {
  const { pendingApproval, approving, decideAction } = useIncidentStore();
  const [note, setNote] = useState('');

  const handle = async (decision) => {
    await decideAction(decision, note.trim() || undefined);
    setNote('');
  };

  if (!pendingApproval) {
    return (
      <Card variant="outlined" sx={{ borderStyle: 'dashed' }}>
        <CardContent sx={{ textAlign: 'center', py: 3 }}>
          <ShieldOutlined sx={{ fontSize: 36, opacity: 0.4, mb: 1 }} />
          <Typography variant="subtitle2" color="text.secondary">
            Approval Gate
          </Typography>
          <Typography variant="body2" color="text.secondary">
            No action is waiting on you. Risky remediations will surface here for approval before
            they execute.
          </Typography>
        </CardContent>
      </Card>
    );
  }

  const { action_type, target, risk, rationale } = pendingApproval;

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
            Approval Required
          </Typography>
          {risk && <Chip size="small" color={riskColor(risk)} label={`${risk} risk`} />}
        </Box>

        <Stack spacing={0.75} sx={{ mb: 1.5 }}>
          <Detail label="Action" value={action_type} mono />
          <Detail label="Target" value={target} mono />
        </Stack>

        {rationale && (
          <Box
            sx={{
              p: 1.5,
              borderRadius: 1.5,
              bgcolor: 'rgba(148, 163, 184, 0.08)',
              mb: 1.5,
            }}
          >
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
              Agent rationale
            </Typography>
            <Typography variant="body2">{rationale}</Typography>
          </Box>
        )}

        <TextField
          fullWidth
          size="small"
          label="Note (optional)"
          placeholder="Add context for the audit log"
          value={note}
          onChange={(e) => setNote(e.target.value)}
          sx={{ mb: 1.5 }}
        />

        <Divider sx={{ mb: 1.5 }} />

        <Stack direction="row" spacing={1.5}>
          <Button
            fullWidth
            variant="contained"
            color="success"
            startIcon={approving ? <CircularProgress size={18} color="inherit" /> : <CheckCircle />}
            disabled={approving}
            onClick={() => handle('approve')}
          >
            Approve
          </Button>
          <Button
            fullWidth
            variant="outlined"
            color="error"
            startIcon={<Cancel />}
            disabled={approving}
            onClick={() => handle('reject')}
          >
            Reject
          </Button>
        </Stack>

        <Collapse in={approving}>
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            Submitting decision to the agent loop...
          </Typography>
        </Collapse>
      </CardContent>
    </Card>
  );
}

function Detail({ label, value, mono }) {
  return (
    <Box sx={{ display: 'flex', gap: 1 }}>
      <Typography variant="body2" color="text.secondary" sx={{ minWidth: 64 }}>
        {label}
      </Typography>
      <Typography
        variant="body2"
        sx={{
          fontFamily: mono ? 'ui-monospace, SFMono-Regular, Menlo, monospace' : undefined,
          wordBreak: 'break-word',
        }}
      >
        {value || '-'}
      </Typography>
    </Box>
  );
}
