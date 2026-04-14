import React, { useEffect, useState } from 'react';
import {
  Box, Paper, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Chip, Button, CircularProgress, Grid, Alert, Dialog, DialogTitle, DialogContent,
  DialogActions, RadioGroup, Radio, FormControlLabel, TextField, Divider, Stack,
} from '@mui/material';
import { Security, CheckCircle, Refresh, Timer, Warning, Gavel, ExpandMore, ExpandLess } from '@mui/icons-material';
import { hitlAPI, claimsAPI } from '../../services/api';
import { fmt } from '../../services/currency';
import useClaimsStore from '../../store/claimsStore';
import useAuthStore from '../../store/authStore';

const PRIORITY_COLORS = { critical: '#DC2626', high: '#F59E0B', normal: '#10B981' };
const PRIORITY_LABELS = { critical: 'CRITICAL', high: 'HIGH', normal: 'NORMAL' };

const DECISIONS = [
  { value: 'approved', label: 'Approve Claim', desc: 'Approve the full settlement amount', color: '#10B981' },
  { value: 'approved_partial', label: 'Partial Approval', desc: 'Approve with reduced settlement', color: '#F59E0B' },
  { value: 'denied', label: 'Deny Claim', desc: 'Deny the claim entirely', color: '#EF4444' },
  { value: 'fraud_investigation', label: 'Flag for Fraud Investigation', desc: 'Escalate to fraud team for deeper investigation', color: '#DC2626' },
  { value: 'pending_documents', label: 'Request Documents', desc: 'Ask claimant for additional documentation', color: '#3B82F6' },
];

const AGENT_LABELS = {
  intake_output: { name: 'Intake Validation', icon: '📋' },
  fraud_output: { name: 'Fraud Detection (CrewAI)', icon: '🔍' },
  damage_output: { name: 'Damage Assessment', icon: '🔧' },
  policy_output: { name: 'Policy Compliance', icon: '📄' },
  settlement_output: { name: 'Settlement Calculation', icon: '💰' },
  evaluation_output: { name: 'Quality Evaluation (LLM Judge)', icon: '⚖️' },
};

function AgentTraceCard({ agentKey, data }) {
  const [open, setOpen] = useState(false);
  const label = AGENT_LABELS[agentKey] || { name: agentKey, icon: '🤖' };
  const confidence = data?.confidence ?? data?.assessment_confidence ?? data?.overall_score;
  const decision = data?.decision?.value || data?.decision || data?.coverage_status?.value || data?.repair_vs_replace;
  const reasoning = data?.intake_notes || data?.crew_summary || data?.assessment_notes || data?.coverage_notes || data?.calculation_breakdown?.join('; ') || '';

  return (
    <Paper sx={{ mb: 1, bgcolor: '#0F172A', borderRadius: 2, border: '1px solid #334155', overflow: 'hidden' }}>
      <Box sx={{ p: 1.5, display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer' }}
        onClick={() => setOpen(!open)}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography fontSize="1.1rem">{label.icon}</Typography>
          <Box>
            <Typography variant="body2" fontWeight={600} color="#F1F5F9">{label.name}</Typography>
            {decision && <Typography variant="caption" color="#94A3B8">Decision: {decision}</Typography>}
          </Box>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {confidence != null && (
            <Chip label={`${(confidence * 100).toFixed(0)}%`} size="small"
              sx={{ fontWeight: 600, bgcolor: confidence >= 0.7 ? '#10B98120' : confidence >= 0.5 ? '#F59E0B20' : '#EF444420',
                color: confidence >= 0.7 ? '#10B981' : confidence >= 0.5 ? '#F59E0B' : '#EF4444' }} />
          )}
          {open ? <ExpandLess sx={{ color: '#64748B' }} /> : <ExpandMore sx={{ color: '#64748B' }} />}
        </Box>
      </Box>
      {open && (
        <Box sx={{ px: 2, pb: 2, borderTop: '1px solid #334155' }}>
          {reasoning && (
            <Box sx={{ mt: 1 }}>
              <Typography variant="caption" color="#64748B" fontWeight={600}>Reasoning</Typography>
              <Typography variant="body2" color="#CBD5E1" sx={{ mt: 0.5 }}>{reasoning}</Typography>
            </Box>
          )}
          {data?.validation_flags?.length > 0 && (
            <Box sx={{ mt: 1 }}>
              <Typography variant="caption" color="#64748B" fontWeight={600}>Flags</Typography>
              {data.validation_flags.map((f, i) => (
                <Chip key={i} label={f} size="small" sx={{ mr: 0.5, mt: 0.5, bgcolor: '#EF444420', color: '#EF4444', fontSize: '0.7rem' }} />
              ))}
            </Box>
          )}
          {data?.primary_concerns?.length > 0 && (
            <Box sx={{ mt: 1 }}>
              <Typography variant="caption" color="#64748B" fontWeight={600}>Concerns</Typography>
              {data.primary_concerns.map((f, i) => (
                <Chip key={i} label={f} size="small" sx={{ mr: 0.5, mt: 0.5, bgcolor: '#EF444420', color: '#EF4444', fontSize: '0.7rem' }} />
              ))}
            </Box>
          )}
          {data?.denial_reasons?.length > 0 && (
            <Box sx={{ mt: 1 }}>
              <Typography variant="caption" color="#64748B" fontWeight={600}>Denial Reasons</Typography>
              {data.denial_reasons.map((r, i) => (
                <Typography key={i} variant="body2" color="#EF4444">• {r}</Typography>
              ))}
            </Box>
          )}
        </Box>
      )}
    </Paper>
  );
}

export default function HITLQueue() {
  const { hitlQueue, hitlStats, hitlLoading, setHITLQueue, setHITLStats, setHITLLoading, removeFromHITL, updateClaimStatus } = useClaimsStore();
  const user = useAuthStore((s) => s.user);
  const [selectedTicket, setSelectedTicket] = useState(null);
  const [fullTicket, setFullTicket] = useState(null);
  const [claimData, setClaimData] = useState(null);
  const [decision, setDecision] = useState('');
  const [notes, setNotes] = useState('');
  const [overrideAI, setOverrideAI] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [successMsg, setSuccessMsg] = useState('');

  const fetchQueue = async () => {
    setHITLLoading(true);
    try {
      const [qRes, sRes] = await Promise.all([hitlAPI.getQueue(), hitlAPI.getStats()]);
      setHITLQueue(qRes.data || []);
      setHITLStats(sRes.data || {});
    } catch {
      setHITLQueue([]);
      setHITLStats({ pending_total: 0, pending_critical: 0, pending_high: 0, resolved_today: 0, human_overrides_today: 0 });
    } finally {
      setHITLLoading(false);
    }
  };

  useEffect(() => { fetchQueue(); }, []);

  const openReview = async (ticket) => {
    setSelectedTicket(ticket);
    setDecision('');
    setNotes('');
    setOverrideAI(false);
    setError('');
    setClaimData(null);
    try {
      const [ticketRes, claimRes] = await Promise.all([
        hitlAPI.getTicket(ticket.ticket_id),
        claimsAPI.getClaimById(ticket.claim_id),
      ]);
      setFullTicket(ticketRes.data);
      setClaimData(claimRes.data);
    } catch {
      setFullTicket(null);
      setClaimData(null);
    }
  };

  const handleSubmitDecision = async () => {
    setSubmitting(true);
    setError('');
    try {
      await hitlAPI.submitDecision(selectedTicket.ticket_id, {
        reviewer_id: user?.username || 'web_reviewer',
        decision,
        notes,
        override_ai: overrideAI,
      });
      const decidedClaim = selectedTicket.claim_id;
      const decidedAction = decision;
      removeFromHITL(selectedTicket.ticket_id);
      updateClaimStatus(decidedClaim, 'processing');
      setSelectedTicket(null);
      setSuccessMsg(`Decision "${decidedAction}" submitted for ${decidedClaim}. Pipeline is resuming — the claim will update shortly.`);
      fetchQueue();
      // Auto-dismiss after 8s
      setTimeout(() => setSuccessMsg(''), 8000);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to submit decision');
    } finally {
      setSubmitting(false);
    }
  };

  const stats = hitlStats || {};

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h5" fontWeight={700} color="#F1F5F9">
            <Security sx={{ mr: 1, verticalAlign: 'middle', color: '#EF4444' }} />
            HITL (Human-In-The-Loop) Review Queue
          </Typography>
          <Typography variant="body2" color="#94A3B8">Claims paused at HITL checkpoint for reviewer approval</Typography>
        </Box>
        <Button startIcon={<Refresh />} onClick={fetchQueue} sx={{ color: '#94A3B8' }}>Refresh</Button>
      </Box>

      {successMsg && <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccessMsg('')}>{successMsg}</Alert>}

      {/* Stats */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {[
          { label: 'Pending Total', value: stats.pending_total || 0, color: '#3B82F6', icon: <Timer /> },
          { label: 'Critical', value: stats.pending_critical || 0, color: '#DC2626', icon: <Warning /> },
          { label: 'High', value: stats.pending_high || 0, color: '#F59E0B', icon: <Warning /> },
          { label: 'Resolved Today', value: stats.resolved_today || 0, color: '#10B981', icon: <CheckCircle /> },
          { label: 'AI Overrides', value: stats.human_overrides_today || 0, color: '#9C27B0', icon: <Gavel /> },
        ].map((s) => (
          <Grid item xs={6} md={2.4} key={s.label}>
            <Paper sx={{ p: 2, bgcolor: '#1E293B', borderRadius: 2, border: '1px solid #334155', display: 'flex', alignItems: 'center', gap: 1.5 }}>
              <Box sx={{ color: s.color }}>{s.icon}</Box>
              <Box>
                <Typography variant="h5" fontWeight={700} sx={{ color: s.color, lineHeight: 1 }}>{s.value}</Typography>
                <Typography variant="caption" color="#94A3B8">{s.label}</Typography>
              </Box>
            </Paper>
          </Grid>
        ))}
      </Grid>

      {/* Queue Table */}
      {hitlLoading ? (
        <Box sx={{ textAlign: 'center', py: 6 }}><CircularProgress /></Box>
      ) : hitlQueue.length === 0 ? (
        <Paper sx={{ p: 6, bgcolor: '#1E293B', borderRadius: 3, textAlign: 'center', border: '1px solid #334155' }}>
          <CheckCircle sx={{ fontSize: 64, color: '#10B981', mb: 2 }} />
          <Typography variant="h6" color="#94A3B8">Queue Empty</Typography>
          <Typography variant="body2" color="#64748B">All claims have been reviewed. Process a high-value claim to populate this queue.</Typography>
        </Paper>
      ) : (
        <TableContainer component={Paper} sx={{ bgcolor: '#1E293B', borderRadius: 3, border: '1px solid #334155' }}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Priority</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Ticket ID</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Claim ID</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Score</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Triggers</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>SLA</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Action</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {hitlQueue.map((ticket) => (
                <TableRow key={ticket.ticket_id} hover sx={{ '&:hover': { bgcolor: '#1E293B80' } }}>
                  <TableCell>
                    <Chip label={PRIORITY_LABELS[ticket.priority] || ticket.priority}
                      sx={{ fontWeight: 700, fontSize: '0.75rem', bgcolor: `${PRIORITY_COLORS[ticket.priority]}20`, color: PRIORITY_COLORS[ticket.priority], border: `1px solid ${PRIORITY_COLORS[ticket.priority]}40` }}
                      size="small"
                    />
                  </TableCell>
                  <TableCell sx={{ color: '#F1F5F9', fontFamily: 'monospace', fontSize: '0.8rem' }}>{ticket.ticket_id}</TableCell>
                  <TableCell sx={{ color: '#94A3B8', fontFamily: 'monospace', fontSize: '0.8rem' }}>{ticket.claim_id}</TableCell>
                  <TableCell sx={{ color: '#F1F5F9', fontWeight: 600 }}>{ticket.priority_score?.toFixed(0)}</TableCell>
                  <TableCell>
                    <Stack spacing={0.5}>
                      {(ticket.triggers || []).slice(0, 2).map((t, i) => (
                        <Typography key={i} variant="caption" color="#F59E0B" sx={{ display: 'block' }}>- {t}</Typography>
                      ))}
                      {(ticket.triggers?.length || 0) > 2 && (
                        <Typography variant="caption" color="#64748B">+{ticket.triggers.length - 2} more</Typography>
                      )}
                    </Stack>
                  </TableCell>
                  <TableCell sx={{ color: '#64748B', fontSize: '0.8rem' }}>{ticket.sla_deadline ? new Date(ticket.sla_deadline).toLocaleString() : '-'}</TableCell>
                  <TableCell>
                    <Button variant="contained" size="small" onClick={() => openReview(ticket)}
                      sx={{ borderRadius: 2, fontWeight: 600, background: 'linear-gradient(135deg, #2563EB, #1D4ED8)', textTransform: 'none' }}>
                      Review
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {/* Review Dialog */}
      <Dialog open={!!selectedTicket} onClose={() => setSelectedTicket(null)} maxWidth="md" fullWidth
        PaperProps={{ sx: { bgcolor: '#1E293B', border: '1px solid #334155', maxHeight: '90vh' } }}>
        <DialogTitle sx={{ color: '#F1F5F9', borderBottom: '1px solid #334155' }}>
          <Security sx={{ mr: 1, verticalAlign: 'middle', color: '#EF4444' }} />
          Review: {selectedTicket?.ticket_id}
          {selectedTicket && (
            <Chip label={PRIORITY_LABELS[selectedTicket.priority]} size="small" sx={{ ml: 2, fontWeight: 700, bgcolor: `${PRIORITY_COLORS[selectedTicket.priority]}20`, color: PRIORITY_COLORS[selectedTicket.priority] }} />
          )}
        </DialogTitle>
        <DialogContent sx={{ mt: 2 }}>
          {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

          {/* Triggers */}
          <Alert severity="warning" sx={{ mb: 2 }}>
            <Typography variant="subtitle2" fontWeight={600}>Review Triggers:</Typography>
            {(selectedTicket?.triggers || []).map((t, i) => (
              <Typography key={i} variant="body2">- {t}</Typography>
            ))}
          </Alert>

          {/* Claim Summary */}
          {claimData && (
            <Paper sx={{ p: 2, mb: 2, bgcolor: '#0F172A', borderRadius: 2, border: '1px solid #334155' }}>
              <Typography variant="subtitle2" color="#94A3B8" fontWeight={600} gutterBottom>Claim Summary</Typography>
              <Grid container spacing={2}>
                <Grid item xs={4}><Typography variant="caption" color="#64748B">Claim ID</Typography><Typography color="#F1F5F9" fontFamily="monospace" fontSize="0.85rem">{claimData.claim_id}</Typography></Grid>
                <Grid item xs={4}><Typography variant="caption" color="#64748B">Type</Typography><Typography color="#F1F5F9">{claimData.incident_type?.replace('_', ' ')}</Typography></Grid>
                <Grid item xs={4}><Typography variant="caption" color="#64748B">Claimed Amount</Typography><Typography color="#F1F5F9" fontWeight={600}>{fmt(claimData.estimated_amount)}</Typography></Grid>
                <Grid item xs={12}><Typography variant="caption" color="#64748B">Description</Typography><Typography color="#94A3B8" fontSize="0.85rem">{claimData.incident_description}</Typography></Grid>
              </Grid>
            </Paper>
          )}

          {/* Agent Traces */}
          {claimData?.agent_outputs && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" color="#94A3B8" fontWeight={600} gutterBottom>
                Agent Analysis (click to expand)
              </Typography>
              {Object.keys(AGENT_LABELS).map((key) =>
                claimData.agent_outputs[key] ? (
                  <AgentTraceCard key={key} agentKey={key} data={claimData.agent_outputs[key]} />
                ) : null
              )}
            </Box>
          )}

          <Divider sx={{ borderColor: '#334155', my: 2 }} />

          {/* Decision */}
          <Typography variant="subtitle2" color="#F1F5F9" fontWeight={600} gutterBottom>Your Decision</Typography>
          <RadioGroup value={decision} onChange={(e) => setDecision(e.target.value)}>
            {DECISIONS.map((d) => (
              <Paper key={d.value} sx={{ p: 1.5, mb: 1, bgcolor: decision === d.value ? `${d.color}15` : '#0F172A', borderRadius: 2, border: `1px solid ${decision === d.value ? d.color : '#334155'}`, cursor: 'pointer' }}
                onClick={() => setDecision(d.value)}>
                <FormControlLabel
                  value={d.value}
                  control={<Radio sx={{ color: '#64748B', '&.Mui-checked': { color: d.color } }} />}
                  label={
                    <Box>
                      <Typography variant="body2" fontWeight={600} color="#F1F5F9">{d.label}</Typography>
                      <Typography variant="caption" color="#94A3B8">{d.desc}</Typography>
                    </Box>
                  }
                />
              </Paper>
            ))}
          </RadioGroup>

          {/* Override checkbox */}
          <FormControlLabel
            control={<input type="checkbox" checked={overrideAI} onChange={(e) => setOverrideAI(e.target.checked)} style={{ marginRight: 8 }} />}
            label={<Typography variant="body2" color="#F59E0B">Override AI recommendation (your decision differs from AI)</Typography>}
            sx={{ mt: 1, mb: 2 }}
          />

          {/* Notes */}
          <TextField fullWidth multiline rows={3} label="Review Notes *" value={notes} onChange={(e) => setNotes(e.target.value)}
            placeholder="Explain your reasoning..." helperText={`${notes.length}/1000 - Required for audit trail`}
            inputProps={{ maxLength: 1000 }}
          />
        </DialogContent>
        <DialogActions sx={{ p: 2, borderTop: '1px solid #334155' }}>
          <Button onClick={() => setSelectedTicket(null)} sx={{ color: '#94A3B8' }}>Cancel</Button>
          <Button variant="contained" onClick={handleSubmitDecision}
            disabled={!decision || submitting}
            sx={{ borderRadius: 2, fontWeight: 600, px: 3, background: 'linear-gradient(135deg, #2563EB, #1D4ED8)' }}>
            {submitting ? <CircularProgress size={20} /> : 'Submit Decision'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
