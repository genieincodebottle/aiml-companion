import React, { useEffect, useState } from 'react';
import {
  Box, Paper, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Chip, Button, CircularProgress, Grid, Tabs, Tab, TextField, InputAdornment,
  IconButton, Tooltip, LinearProgress, Stack, Dialog, DialogTitle, DialogContent, DialogActions, Alert,
} from '@mui/material';
import { Search, Refresh, Visibility, Close, PlayArrow, Gavel, ExpandMore, ExpandLess, CheckCircle, Warning, Error as ErrorIcon, Psychology } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { claimsAPI } from '../../services/api';
import { fmt } from '../../services/currency';

const STATUS_COLORS = {
  approved: 'success', approved_partial: 'warning', denied: 'error', auto_rejected: 'error',
  escalated_human_review: 'warning', fraud_investigation: 'error', pending: 'info', processing: 'info',
  pending_human_review: 'warning', submitted: 'info', failed: 'error',
};

const ACTIVE_STATUSES = ['processing', 'submitted', 'pending_human_review'];

export default function ClaimsDashboard() {
  const navigate = useNavigate();
  const [claims, setClaims] = useState([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState(0);
  const [search, setSearch] = useState('');
  const [detail, setDetail] = useState(null);         // full claim for dialog
  const [detailLoading, setDetailLoading] = useState(false);
  const [actionMsg, setActionMsg] = useState(null);   // {type,text} for dialog alert

  const fetchClaims = async () => {
    setLoading(true);
    try {
      const res = await claimsAPI.getAllClaims(null, 200);
      setClaims(res.data.claims || res.data || []);
    } catch {
      setClaims([]);
    } finally {
      setLoading(false);
    }
  };

  const openDetail = async (claimId) => {
    setActionMsg(null);
    setDetailLoading(true);
    setDetail({ claim_id: claimId, _loading: true });
    try {
      const res = await claimsAPI.getClaimById(claimId);
      setDetail(res.data);
    } catch (err) {
      setDetail({ claim_id: claimId, _error: err.response?.data?.detail || 'Failed to load claim' });
    } finally {
      setDetailLoading(false);
    }
  };

  const refreshDetail = async () => {
    if (!detail?.claim_id) return;
    try {
      const res = await claimsAPI.getClaimById(detail.claim_id);
      setDetail(res.data);
    } catch { /* keep existing */ }
  };

  const handleProcess = async () => {
    if (!detail?.claim_id) return;
    setActionMsg(null);
    try {
      await claimsAPI.processClaim(detail.claim_id);
      setActionMsg({ type: 'success', text: 'Pipeline started. Refreshing in 3 seconds…' });
      setTimeout(async () => {
        await refreshDetail();
        await fetchClaims();
      }, 3000);
    } catch (err) {
      setActionMsg({ type: 'error', text: err.response?.data?.detail || 'Failed to start pipeline' });
    }
  };

  const handleGoToHITL = () => {
    setDetail(null);
    navigate('/hitl-queue');
  };

  useEffect(() => { fetchClaims(); }, []);

  // Auto-poll while any claim is actively processing
  useEffect(() => {
    const hasActive = claims.some((c) => ACTIVE_STATUSES.includes(c.status));
    if (!hasActive) return;
    const interval = setInterval(fetchClaims, 5000);
    return () => clearInterval(interval);
  }, [claims]);

  const TAB_FILTERS = ['all', 'pending', 'escalated_human_review', 'approved', 'denied'];
  const TAB_LABELS = ['All', 'Pending', 'HITL (Human-In-The-Loop) / Flagged', 'Approved', 'Denied'];

  const filtered = claims.filter((c) => {
    const matchTab = tab === 0 || c.status === TAB_FILTERS[tab] ||
      (tab === 2 && ['escalated_human_review', 'fraud_investigation', 'pending_human_review'].includes(c.status)) ||
      (tab === 4 && ['denied', 'auto_rejected'].includes(c.status));
    const matchSearch = !search || JSON.stringify(c).toLowerCase().includes(search.toLowerCase());
    return matchTab && matchSearch;
  });

  const stats = {
    total: claims.length,
    pending: claims.filter((c) => ['pending', 'processing', 'submitted'].includes(c.status)).length,
    hitl: claims.filter((c) => ['escalated_human_review', 'fraud_investigation', 'pending_human_review'].includes(c.status)).length,
    approved: claims.filter((c) => c.status?.includes('approved')).length,
    denied: claims.filter((c) => ['denied', 'auto_rejected'].includes(c.status)).length,
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h5" fontWeight={700} color="#F1F5F9">Claims Dashboard</Typography>
          <Typography variant="body2" color="#94A3B8">Review and manage insurance claims</Typography>
        </Box>
        <Button startIcon={<Refresh />} onClick={fetchClaims} sx={{ color: '#94A3B8' }}>Refresh</Button>
      </Box>

      {/* Stats */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {[
          { label: 'Total Claims', value: stats.total, color: '#3B82F6' },
          { label: 'Needs Review', value: stats.pending, color: '#F59E0B' },
          { label: 'HITL Flagged', value: stats.hitl, color: '#EF4444' },
          { label: 'Approved', value: stats.approved, color: '#10B981' },
          { label: 'Denied', value: stats.denied, color: '#DC2626' },
        ].map((s) => (
          <Grid item xs={6} md={2.4} key={s.label}>
            <Paper sx={{ p: 2, bgcolor: '#1E293B', borderRadius: 2, border: '1px solid #334155', textAlign: 'center' }}>
              <Typography variant="h4" fontWeight={700} sx={{ color: s.color }}>{s.value}</Typography>
              <Typography variant="caption" color="#94A3B8">{s.label}</Typography>
            </Paper>
          </Grid>
        ))}
      </Grid>

      {/* Search + Tabs */}
      <TextField
        fullWidth placeholder="Search claims by ID, description, type..." value={search}
        onChange={(e) => setSearch(e.target.value)} sx={{ mb: 2 }}
        InputProps={{ startAdornment: <InputAdornment position="start"><Search sx={{ color: '#64748B' }} /></InputAdornment> }}
      />
      <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ mb: 2, '& .MuiTab-root': { color: '#94A3B8', fontWeight: 600, minWidth: 100 }, '& .Mui-selected': { color: '#3B82F6' } }}>
        {TAB_LABELS.map((l, i) => (
          <Tab key={l} label={<Stack direction="row" spacing={0.5} alignItems="center">
            <span>{l}</span>
            {i > 0 && <Chip label={[0, stats.pending, stats.hitl, stats.approved, stats.denied][i]} size="small" sx={{ height: 20, fontSize: '0.7rem' }} />}
          </Stack>} />
        ))}
      </Tabs>

      {/* Table */}
      {loading ? (
        <Box sx={{ textAlign: 'center', py: 6 }}><CircularProgress /></Box>
      ) : (
        <TableContainer component={Paper} sx={{ bgcolor: '#1E293B', borderRadius: 3, border: '1px solid #334155' }}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Claim ID</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Type</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Amount</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Fraud Score</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Status</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Settlement</TableCell>
                <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filtered.map((claim) => (
                <TableRow key={claim.claim_id} hover sx={{ '&:hover': { bgcolor: '#1E293B80' } }}>
                  <TableCell sx={{ color: '#F1F5F9', fontFamily: 'monospace', fontSize: '0.8rem' }}>{claim.claim_id}</TableCell>
                  <TableCell sx={{ color: '#94A3B8', fontSize: '0.85rem' }}>{claim.incident_type?.replace(/_/g, ' ')}</TableCell>
                  <TableCell sx={{ color: '#F1F5F9', fontWeight: 600 }}>{fmt(claim.estimated_amount)}</TableCell>
                  <TableCell>
                    {claim.fraud_score != null ? (
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <LinearProgress
                          variant="determinate" value={claim.fraud_score * 100}
                          sx={{ width: 60, height: 6, borderRadius: 3, bgcolor: '#334155',
                            '& .MuiLinearProgress-bar': { bgcolor: claim.fraud_score > 0.65 ? '#EF4444' : claim.fraud_score > 0.35 ? '#F59E0B' : '#10B981' }
                          }}
                        />
                        <Typography variant="caption" color="#94A3B8">{(claim.fraud_score * 100).toFixed(0)}%</Typography>
                      </Box>
                    ) : '-'}
                  </TableCell>
                  <TableCell>
                    <Chip label={claim.status?.replace(/_/g, ' ')} color={STATUS_COLORS[claim.status] || 'default'} size="small" sx={{ fontWeight: 600, fontSize: '0.75rem' }} />
                  </TableCell>
                  <TableCell sx={{ color: '#10B981', fontWeight: 600, fontSize: '0.85rem' }}>
                    {fmt(claim.settlement_amount)}
                  </TableCell>
                  <TableCell>
                    <Tooltip title="View Details">
                      <IconButton size="small" sx={{ color: '#3B82F6' }} onClick={() => openDetail(claim.claim_id)}>
                        <Visibility fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </TableCell>
                </TableRow>
              ))}
              {filtered.length === 0 && (
                <TableRow>
                  <TableCell colSpan={7} sx={{ textAlign: 'center', py: 4, color: '#64748B' }}>
                    No claims found{search ? ` matching "${search}"` : ''}
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {/* Claim Detail Dialog */}
      <ClaimDetailDialog
        claim={detail}
        loading={detailLoading}
        actionMsg={actionMsg}
        onClose={() => { setDetail(null); setActionMsg(null); }}
        onProcess={handleProcess}
        onGoToHITL={handleGoToHITL}
        onRefresh={refreshDetail}
      />
    </Box>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Claim detail dialog - shows every agent's output + pipeline path + costs.
// ─────────────────────────────────────────────────────────────────────────────

function Row({ label, value, mono, color }) {
  if (value === undefined || value === null || value === '') return null;
  return (
    <Box sx={{ display: 'flex', py: 0.75, borderBottom: '1px solid #1E293B' }}>
      <Typography variant="caption" sx={{ color: '#64748B', width: 200, textTransform: 'uppercase', letterSpacing: 0.4 }}>{label}</Typography>
      <Typography variant="body2" sx={{ color: color || '#F1F5F9', fontFamily: mono ? 'monospace' : 'inherit', fontSize: mono ? '0.8rem' : '0.875rem', flex: 1, wordBreak: 'break-word' }}>
        {value}
      </Typography>
    </Box>
  );
}

function SectionTitle({ children }) {
  return (
    <Typography variant="overline" sx={{ color: '#3B82F6', fontWeight: 700, letterSpacing: 1, mt: 2, display: 'block' }}>
      {children}
    </Typography>
  );
}

// Given a claim's status, return {severity, title, body, action} describing
// the next step a reviewer should take. null = no action needed (completed).
function nextStepFor(claim) {
  if (!claim) return null;
  switch (claim.status) {
    case 'submitted':
      return {
        severity: 'warning',
        title: 'Pipeline not started yet',
        body: 'This claim was submitted but the agent pipeline has not run. Click "Process Now" to run it manually. If this keeps happening, check the backend log for an LLM API key error.',
        action: 'process',
      };
    case 'processing':
      return {
        severity: 'info',
        title: 'Agents running…',
        body: 'The LangGraph pipeline is currently executing. Click Refresh in a few seconds to see the updated state.',
        action: 'refresh',
      };
    case 'pending_human_review':
      return {
        severity: 'warning',
        title: 'Awaiting your approval',
        body: 'The pipeline paused at the HITL checkpoint. Open the HITL Review Queue to approve or deny this claim - that will resume the pipeline.',
        action: 'hitl',
      };
    case 'failed':
      return {
        severity: 'error',
        title: 'Pipeline failed',
        body: 'See the Errors section below for details. You can click "Process Now" to retry once the underlying issue is fixed.',
        action: 'process',
      };
    case 'completed':
      return null;
    default:
      return null;
  }
}

// ── Agent names and icons ────────────────────────────────────────────────────
const AGENT_DISPLAY = {
  intake_agent: { label: 'Intake Validation', icon: '📋', color: '#3B82F6' },
  fraud_crew: { label: 'Fraud Detection (CrewAI)', icon: '🔍', color: '#EF4444' },
  damage_assessor: { label: 'Damage Assessment', icon: '🔧', color: '#F59E0B' },
  policy_checker: { label: 'Policy Compliance', icon: '📜', color: '#8B5CF6' },
  settlement_calculator: { label: 'Settlement Calculation', icon: '💰', color: '#10B981' },
  evaluator: { label: 'Quality Evaluation (LLM Judge)', icon: '⚖️', color: '#06B6D4' },
  communication_agent: { label: 'Communication', icon: '📧', color: '#64748B' },
  auto_reject: { label: 'Auto Reject', icon: '🚫', color: '#EF4444' },
  hitl_checkpoint: { label: 'HITL Review', icon: '👤', color: '#F59E0B' },
};

const confidenceColor = (c) => {
  if (c == null) return '#64748B';
  if (c >= 0.75) return '#10B981';
  if (c >= 0.60) return '#F59E0B';
  return '#EF4444';
};

const confidenceLabel = (c) => {
  if (c == null) return 'N/A';
  if (c >= 0.75) return 'High';
  if (c >= 0.60) return 'Medium';
  return 'Low';
};

function AgentTracePanel({ agentOutputs }) {
  const [expanded, setExpanded] = useState({});
  const trace = agentOutputs?._trace || [];

  if (!trace.length) return null;

  const toggle = (i) => setExpanded((prev) => ({ ...prev, [i]: !prev[i] }));

  return (
    <Box sx={{ mt: 1 }}>
      {trace.map((entry, i) => {
        const info = AGENT_DISPLAY[entry.agent] || { label: entry.agent, icon: '⚙️', color: '#64748B' };
        const conf = entry.confidence;
        const isOpen = expanded[i];
        const confPct = conf != null ? `${(conf * 100).toFixed(0)}%` : null;

        return (
          <Paper key={i} sx={{
            mb: 1, bgcolor: '#0F172A', border: `1px solid ${isOpen ? info.color + '60' : '#1E293B'}`,
            borderRadius: 2, overflow: 'hidden', transition: 'border-color 0.2s',
          }}>
            {/* Header - always visible */}
            <Box
              onClick={() => toggle(i)}
              sx={{
                display: 'flex', alignItems: 'center', gap: 1.5, px: 2, py: 1.5,
                cursor: 'pointer', '&:hover': { bgcolor: '#1E293B40' },
              }}
            >
              <Typography sx={{ fontSize: '1.2rem', lineHeight: 1 }}>{info.icon}</Typography>
              <Box sx={{ flex: 1, minWidth: 0 }}>
                <Typography variant="body2" fontWeight={700} color="#F1F5F9" noWrap>
                  {info.label}
                </Typography>
                {entry.decision && (
                  <Typography variant="caption" color="#94A3B8">
                    Decision: <span style={{ color: '#F1F5F9', fontWeight: 600 }}>{entry.decision}</span>
                  </Typography>
                )}
              </Box>

              {/* Confidence bar */}
              {confPct && (
                <Box sx={{ width: 100, textAlign: 'right' }}>
                  <Typography variant="caption" sx={{ color: confidenceColor(conf), fontWeight: 700 }}>
                    {confPct} {confidenceLabel(conf)}
                  </Typography>
                  <LinearProgress variant="determinate" value={(conf || 0) * 100}
                    sx={{
                      height: 4, borderRadius: 2, bgcolor: '#334155', mt: 0.5,
                      '& .MuiLinearProgress-bar': { bgcolor: confidenceColor(conf), borderRadius: 2 },
                    }}
                  />
                </Box>
              )}

              {/* Duration */}
              {entry.duration_ms != null && (
                <Typography variant="caption" color="#64748B" sx={{ minWidth: 50, textAlign: 'right' }}>
                  {entry.duration_ms > 1000 ? `${(entry.duration_ms / 1000).toFixed(1)}s` : `${entry.duration_ms}ms`}
                </Typography>
              )}

              {/* Flags count badge */}
              {entry.flags?.length > 0 && (
                <Chip label={`${entry.flags.length} flag${entry.flags.length > 1 ? 's' : ''}`}
                  size="small" sx={{ bgcolor: '#7F1D1D', color: '#FCA5A5', height: 20, fontSize: '0.7rem' }} />
              )}

              {isOpen ? <ExpandLess sx={{ color: '#64748B' }} /> : <ExpandMore sx={{ color: '#64748B' }} />}
            </Box>

            {/* Expanded body */}
            {isOpen && (
              <Box sx={{ px: 2, pb: 2, borderTop: '1px solid #1E293B' }}>
                {/* Reasoning */}
                {entry.reasoning && (
                  <Box sx={{ mt: 1.5 }}>
                    <Typography variant="caption" color="#64748B" fontWeight={600}>Reasoning</Typography>
                    <Typography variant="body2" color="#CBD5E1" sx={{ mt: 0.5, whiteSpace: 'pre-line', fontSize: '0.85rem' }}>
                      {entry.reasoning}
                    </Typography>
                  </Box>
                )}

                {/* Flags */}
                {entry.flags?.length > 0 && (
                  <Box sx={{ mt: 1.5 }}>
                    <Typography variant="caption" color="#64748B" fontWeight={600}>Flags</Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                      {entry.flags.map((flag, j) => (
                        <Chip key={j} label={flag} size="small"
                          sx={{ bgcolor: '#7F1D1D', color: '#FCA5A5', fontSize: '0.75rem', height: 22 }} />
                      ))}
                    </Box>
                  </Box>
                )}

                {/* Key Findings */}
                {entry.findings && (
                  <Box sx={{ mt: 1.5 }}>
                    <Typography variant="caption" color="#64748B" fontWeight={600}>Key Findings</Typography>
                    <Box sx={{ mt: 0.5, display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '2px 12px' }}>
                      {Object.entries(entry.findings).map(([k, v]) => (
                        <React.Fragment key={k}>
                          <Typography variant="caption" color="#64748B">{k.replace(/_/g, ' ')}</Typography>
                          <Typography variant="caption" color="#F1F5F9" fontWeight={600}>
                            {typeof v === 'boolean' ? (v ? 'Yes' : 'No') :
                             typeof v === 'number' ? (Number.isInteger(v) ? v : v.toLocaleString(undefined, { maximumFractionDigits: 2 })) :
                             Array.isArray(v) ? (v.length ? v.join(', ') : 'None') :
                             String(v ?? '-')}
                          </Typography>
                        </React.Fragment>
                      ))}
                    </Box>
                  </Box>
                )}
              </Box>
            )}
          </Paper>
        );
      })}
    </Box>
  );
}

function ClaimDetailDialog({ claim, loading, actionMsg, onClose, onProcess, onGoToHITL, onRefresh }) {
  const open = !!claim;
  const money = fmt;
  const pct = (v) => (v == null ? '-' : `${(v * 100).toFixed(1)}%`);
  const pipelinePath = Array.isArray(claim?.pipeline_path) ? claim.pipeline_path : [];
  const errors = Array.isArray(claim?.error_log) ? claim.error_log : [];
  const step = claim && !claim._loading && !claim._error ? nextStepFor(claim) : null;

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth
      PaperProps={{ sx: { bgcolor: '#0F172A', border: '1px solid #334155', borderRadius: 3 } }}>
      <DialogTitle sx={{ color: '#F1F5F9', borderBottom: '1px solid #1E293B', display: 'flex', justifyContent: 'space-between', alignItems: 'center', pb: 1.5 }}>
        <Box>
          <Typography variant="h6" fontWeight={700}>Claim Details</Typography>
          {claim && !claim._loading && !claim._error && (
            <Typography variant="caption" sx={{ color: '#64748B', fontFamily: 'monospace' }}>{claim.claim_id}</Typography>
          )}
        </Box>
        <IconButton onClick={onClose} size="small" sx={{ color: '#94A3B8' }}><Close /></IconButton>
      </DialogTitle>

      <DialogContent sx={{ pt: 2 }}>
        {loading || claim?._loading ? (
          <Box sx={{ textAlign: 'center', py: 6 }}><CircularProgress /></Box>
        ) : claim?._error ? (
          <Typography color="error" sx={{ py: 2 }}>{claim._error}</Typography>
        ) : claim ? (
          <>
            {/* Next-step banner - always tells the reviewer what to do */}
            {step && (
              <Alert severity={step.severity} sx={{ mb: 2, bgcolor: '#1E293B', border: `1px solid ${step.severity === 'error' ? '#7F1D1D' : step.severity === 'warning' ? '#78350F' : '#1E3A5F'}`, '& .MuiAlert-message': { color: '#F1F5F9' } }}>
                <Typography variant="subtitle2" fontWeight={700}>{step.title}</Typography>
                <Typography variant="body2" sx={{ color: '#CBD5E1', mt: 0.5 }}>{step.body}</Typography>
              </Alert>
            )}
            {!step && ['completed', 'approved', 'approved_partial', 'denied', 'auto_rejected'].includes(claim.status) && (
              <Alert severity={['denied', 'auto_rejected'].includes(claim.status) ? 'error' : claim.status === 'approved_partial' ? 'warning' : 'success'} sx={{ mb: 2, bgcolor: '#1E293B', border: `1px solid ${['denied', 'auto_rejected'].includes(claim.status) ? '#7F1D1D' : '#064E3B'}`, '& .MuiAlert-message': { color: '#F1F5F9' } }}>
                <Typography variant="subtitle2" fontWeight={700}>Claim complete</Typography>
                <Typography variant="body2" sx={{ color: '#CBD5E1', mt: 0.5 }}>
                  Final decision: <strong>{(claim.final_decision || '').replace(/_/g, ' ')}</strong>. No further action needed.
                </Typography>
                {claim.denial_reasons?.length > 0 && (
                  <Typography variant="body2" sx={{ color: '#FCA5A5', mt: 0.5 }}>
                    Reasons: {claim.denial_reasons.join('; ')}
                  </Typography>
                )}
              </Alert>
            )}
            {actionMsg && (
              <Alert severity={actionMsg.type} sx={{ mb: 2 }}>{actionMsg.text}</Alert>
            )}

            <SectionTitle>Claim</SectionTitle>
            <Row label="Policy Number" value={claim.policy_number} mono />
            <Row label="Incident Type" value={claim.incident_type?.replace(/_/g, ' ')} />
            <Row label="Incident Date" value={claim.incident_date} />
            <Row label="Location" value={claim.incident_location} />
            <Row label="Description" value={claim.incident_description} />
            <Row label="Estimated Amount" value={money(claim.estimated_amount)} />
            <Row label="Created" value={claim.created_at} mono />

            <SectionTitle>Status</SectionTitle>
            <Row label="Status" value={<Chip label={(claim.status || '').replace(/_/g, ' ')} size="small" color={STATUS_COLORS[claim.status] || 'default'} sx={{ fontWeight: 600 }} />} />
            <Row label="Final Decision" value={claim.final_decision?.replace(/_/g, ' ')} color="#F1F5F9" />
            <Row label="Settlement Amount" value={money(claim.settlement_amount)} color="#10B981" />
            {claim.denial_reasons?.length > 0 && (
              <Row label="Denial Reasons" value={claim.denial_reasons.join('; ')} color="#EF4444" />
            )}
            <Row label="Decided By" value={claim.decided_by || (claim.hitl_required ? '-' : 'AI Agent')} color="#60A5FA" />
            <Row label="Decision Date" value={claim.decided_at} mono />
            <Row label="Completed" value={claim.completed_at} mono />

            <SectionTitle>Agent Outputs</SectionTitle>
            <Row
              label="Fraud Score"
              value={claim.fraud_score != null ? `${pct(claim.fraud_score)}  ·  ${claim.fraud_risk_level || 'unknown'}` : '-'}
              color={claim.fraud_score > 0.65 ? '#EF4444' : claim.fraud_score > 0.35 ? '#F59E0B' : '#10B981'}
            />
            <Row
              label="Evaluator Score"
              value={claim.evaluation_score != null ? pct(claim.evaluation_score) : '-'}
              color={claim.evaluation_score >= 0.7 ? '#10B981' : '#F59E0B'}
            />
            <Row label="HITL Required" value={claim.hitl_required ? 'Yes' : 'No'} color={claim.hitl_required ? '#F59E0B' : '#94A3B8'} />
            <Row label="HITL Ticket" value={claim.hitl_ticket_id} mono />

            <SectionTitle>Pipeline</SectionTitle>
            <Row
              label="Path"
              value={pipelinePath.length ? pipelinePath.join('  ->  ') : '-'}
              mono
            />
            <Row label="Agent Calls" value={claim.agent_call_count} />
            <Row label="Total LLM Cost" value={fmt(claim.total_cost_usd)} />
            <Row label="Processing Time" value={claim.processing_time_sec != null ? `${parseFloat(claim.processing_time_sec).toFixed(2)}s` : '-'} />

            {/* Per-Agent Trace */}
            {claim.agent_outputs?._trace?.length > 0 && (
              <>
                <SectionTitle>Agent Trace</SectionTitle>
                <Typography variant="caption" color="#64748B" sx={{ mb: 1, display: 'block' }}>
                  Click each agent to see its reasoning, confidence, findings, and flags.
                </Typography>
                <AgentTracePanel agentOutputs={claim.agent_outputs} />
              </>
            )}

            {errors.length > 0 && (
              <>
                <SectionTitle>Errors</SectionTitle>
                {errors.map((err, i) => (
                  <Typography key={i} variant="body2" sx={{ color: '#EF4444', fontFamily: 'monospace', fontSize: '0.8rem', py: 0.5 }}>
                    · {err}
                  </Typography>
                ))}
              </>
            )}
          </>
        ) : null}
      </DialogContent>

      <DialogActions sx={{ borderTop: '1px solid #1E293B', px: 3, py: 2, gap: 1 }}>
        <Button onClick={onClose} sx={{ color: '#94A3B8' }}>Close</Button>
        {step?.action === 'refresh' && (
          <Button onClick={onRefresh} variant="outlined" startIcon={<Refresh />}>Refresh</Button>
        )}
        {step?.action === 'process' && (
          <Button onClick={onProcess} variant="contained" startIcon={<PlayArrow />} sx={{ bgcolor: '#3B82F6' }}>
            Process Now
          </Button>
        )}
        {step?.action === 'hitl' && (
          <Button onClick={onGoToHITL} variant="contained" startIcon={<Gavel />} sx={{ bgcolor: '#F59E0B', '&:hover': { bgcolor: '#D97706' } }}>
            Open HITL Ticket
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
}
