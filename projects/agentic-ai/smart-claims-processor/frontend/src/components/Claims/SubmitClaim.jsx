import React, { useEffect, useState } from 'react';
import {
  Box, Paper, Typography, TextField, Button, Grid, MenuItem, Alert,
  CircularProgress, Divider, Chip, Collapse, IconButton, Tooltip,
} from '@mui/material';
import { Send, Add, ExpandMore, ExpandLess, CheckCircle, Policy } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { claimsAPI, policyAPI } from '../../services/api';
import api from '../../services/api';
import useAuthStore from '../../store/authStore';

// ── Quick-create panel defaults ────────────────────────────────────────────

function randomSuffix() {
  return String(Math.floor(100000 + Math.random() * 900000));
}

export default function SubmitClaim() {
  const navigate = useNavigate();
  const user = useAuthStore((s) => s.user);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState('');

  // Country-aware state
  const [claimTypes, setClaimTypes] = useState([]);
  const [currencySymbol, setCurrencySymbol] = useState('$');
  const [countryName, setCountryName] = useState('');
  const [countryCode, setCountryCode] = useState('US');

  // Existing policies list
  const [policies, setPolicies] = useState([]);
  const [policiesLoading, setPoliciesLoading] = useState(true);

  // Quick-create policy state
  const [templates, setTemplates] = useState([]);
  const [showQuickCreate, setShowQuickCreate] = useState(false);
  const [quickForm, setQuickForm] = useState(null);
  const [quickSaving, setQuickSaving] = useState(false);
  const [quickMsg, setQuickMsg] = useState(null);

  // Claim form
  const [form, setForm] = useState({
    policy_number: '',
    incident_type: '',
    incident_date: new Date().toISOString().split('T')[0],
    incident_description: '',
    incident_location: '',
    police_report_number: '',
    estimated_amount: '',
    vehicle_year: '',
    vehicle_make: '',
    vehicle_model: '',
  });

  // Load country settings + policies + templates on mount
  useEffect(() => {
    // Country settings
    (async () => {
      try {
        const res = await api.get('/api/settings/country');
        const { claim_types, active } = res.data;
        setClaimTypes((claim_types || []).map((t) => ({
          value: t,
          label: t.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
        })));
        setCurrencySymbol(active?.currency_symbol || '$');
        setCountryName(active?.name || '');
        setCountryCode(active?.code || 'US');
        if (claim_types?.length) {
          setForm((f) => ({ ...f, incident_type: claim_types[0] }));
        }
      } catch {
        setClaimTypes([
          { value: 'auto_collision', label: 'Auto Collision' },
          { value: 'auto_theft', label: 'Auto Theft' },
          { value: 'property_fire', label: 'Property Fire' },
          { value: 'liability', label: 'Liability' },
        ]);
        setForm((f) => ({ ...f, incident_type: 'auto_collision' }));
      }
    })();

    // Existing policies
    (async () => {
      setPoliciesLoading(true);
      try {
        const res = await policyAPI.list();
        setPolicies(res.data || []);
      } catch {
        setPolicies([]);
      } finally {
        setPoliciesLoading(false);
      }
    })();

    // Policy templates (country-aware defaults)
    (async () => {
      try {
        const res = await policyAPI.defaults();
        setTemplates(res.data?.templates || []);
      } catch {
        setTemplates([]);
      }
    })();
  }, []);

  // ── Quick-create helpers ─────────────────────────────────────────────────

  const startQuickCreate = (tmpl) => {
    setQuickForm({
      ...tmpl,
      policy_number: tmpl.policy_number + randomSuffix(),
      holder_name: user?.username || '',
    });
    setQuickMsg(null);
  };

  const handleQuickSave = async () => {
    if (!quickForm) return;
    setQuickSaving(true);
    setQuickMsg(null);
    try {
      // Remove the UI-only "label" field before sending
      const { label, ...payload } = quickForm;
      await policyAPI.upsert(payload);
      // Refresh policies list and auto-select the new one
      const res = await policyAPI.list();
      setPolicies(res.data || []);
      setForm((f) => ({ ...f, policy_number: quickForm.policy_number }));
      setQuickMsg({ type: 'success', text: `Policy ${quickForm.policy_number} created and selected!` });
      setQuickForm(null);
      // Auto-collapse after a beat
      setTimeout(() => setShowQuickCreate(false), 1500);
    } catch (err) {
      setQuickMsg({ type: 'error', text: err.response?.data?.detail || 'Failed to create policy' });
    } finally {
      setQuickSaving(false);
    }
  };

  const qf = (k, v) => setQuickForm((prev) => ({ ...prev, [k]: v }));

  // ── Claim form helpers ───────────────────────────────────────────────────

  const selectedPolicy = policies.find((p) => p.policy_number === form.policy_number);
  const isAuto = form.incident_type.includes('auto') || form.incident_type === 'own_damage' || form.incident_type === 'theft';

  // Country-aware placeholder hints
  const hints = countryCode === 'IN' ? {
    location: 'e.g. Bengaluru, Mumbai, Delhi',
    policeReport: 'e.g. FIR-2026-12345',
    amount: 'e.g. 50000',
    vehicleYear: 'e.g. 2022',
    vehicleMake: 'e.g. Maruti Suzuki, Tata, Hyundai',
    vehicleModel: 'e.g. Swift, Nexon, Creta',
    description: 'Describe what happened - e.g. "Vehicle hit by truck on NH-44 near Bengaluru toll plaza"',
  } : {
    location: 'e.g. Los Angeles, CA',
    policeReport: 'e.g. PR-2026-12345',
    amount: 'e.g. 5000',
    vehicleYear: 'e.g. 2022',
    vehicleMake: 'e.g. Toyota, Honda, Ford',
    vehicleModel: 'e.g. Camry, Civic, F-150',
    description: 'Describe what happened - e.g. "Rear-ended at intersection on I-405"',
  };
  const canSubmit = form.policy_number && form.incident_type && form.incident_description.length >= 10 && parseFloat(form.estimated_amount) > 0;

  const handleChange = (field) => (e) => setForm({ ...form, [field]: e.target.value });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const payload = {
        ...form,
        estimated_amount: parseFloat(form.estimated_amount),
        vehicle_year: form.vehicle_year ? parseInt(form.vehicle_year) : null,
        documents: ['damage_photos.zip'],
      };
      await claimsAPI.submitClaim(payload);
      setSuccess(true);
      setTimeout(() => navigate('/my-claims'), 2000);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to submit claim');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h5" fontWeight={700} color="#F1F5F9">Submit New Claim</Typography>
          <Typography variant="body2" color="#94A3B8">
            Select a policy (or quick-create one), then fill in your claim details.
          </Typography>
        </Box>
        {countryName && (
          <Typography variant="caption" color="#64748B" sx={{ border: '1px solid #334155', borderRadius: 2, px: 1.5, py: 0.5 }}>
            {countryName} ({currencySymbol})
          </Typography>
        )}
      </Box>

      {success && (
        <Alert severity="success" sx={{ mb: 3 }}>
          Claim submitted successfully! Redirecting to your claims...
        </Alert>
      )}
      {error && <Alert severity="error" sx={{ mb: 3 }}>{error}</Alert>}

      {/* ── Step 1: Policy Selection ──────────────────────────────────────── */}
      <Paper sx={{ p: 3, mb: 3, bgcolor: '#1E293B', borderRadius: 3, border: '1px solid #334155' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Policy sx={{ color: '#3B82F6' }} />
            <Typography variant="subtitle1" fontWeight={700} color="#F1F5F9">
              Step 1: Select Your Insurance Policy
            </Typography>
          </Box>
          {selectedPolicy && (
            <Chip icon={<CheckCircle />} label={`${selectedPolicy.policy_number} - ${selectedPolicy.holder_name}`}
              color="success" size="small" sx={{ fontWeight: 600 }} />
          )}
        </Box>

        {policiesLoading ? (
          <Box sx={{ textAlign: 'center', py: 2 }}><CircularProgress size={24} /></Box>
        ) : (
          <>
            {/* Existing policies dropdown */}
            <TextField select fullWidth label="Choose an existing policy" value={form.policy_number}
              onChange={handleChange('policy_number')} sx={{ mb: 2 }}
              helperText={policies.length === 0 ? 'No policies found - create one below' : `${policies.length} policies available`}>
              <MenuItem value="">-- Select a policy --</MenuItem>
              {policies.map((p) => (
                <MenuItem key={p.policy_number} value={p.policy_number}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                    <Typography fontFamily="monospace" fontWeight={600}>{p.policy_number}</Typography>
                    <Typography variant="body2" color="text.secondary">- {p.holder_name}</Typography>
                    <Chip label={p.type} size="small" sx={{ ml: 'auto', height: 20, fontSize: '0.7rem' }} />
                    <Chip label={p.status} size="small" color={p.status === 'active' ? 'success' : 'error'}
                      sx={{ height: 20, fontSize: '0.7rem' }} />
                  </Box>
                </MenuItem>
              ))}
            </TextField>

            {/* Quick-create toggle */}
            <Divider sx={{ my: 2, borderColor: '#334155' }}>
              <Chip label="OR" size="small" sx={{ color: '#64748B', fontSize: '0.7rem' }} />
            </Divider>

            <Button
              onClick={() => setShowQuickCreate(!showQuickCreate)}
              startIcon={showQuickCreate ? <ExpandLess /> : <Add />}
              endIcon={!showQuickCreate ? <ExpandMore /> : null}
              sx={{ color: '#3B82F6', fontWeight: 600, mb: 1 }}
            >
              Quick-Create a Policy with {countryName || 'Default'} Defaults
            </Button>

            <Collapse in={showQuickCreate}>
              {quickMsg && <Alert severity={quickMsg.type} sx={{ mb: 2 }} onClose={() => setQuickMsg(null)}>{quickMsg.text}</Alert>}

              {/* Template buttons */}
              {!quickForm && (
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 2, mt: 1 }}>
                  {templates.map((tmpl, i) => (
                    <Button key={i} variant="outlined" onClick={() => startQuickCreate(tmpl)}
                      sx={{ borderColor: '#334155', color: '#F1F5F9', textTransform: 'none', borderRadius: 2,
                        '&:hover': { borderColor: '#3B82F6', bgcolor: '#3B82F620' } }}>
                      <Box sx={{ textAlign: 'left' }}>
                        <Typography variant="body2" fontWeight={600}>{tmpl.label}</Typography>
                        <Typography variant="caption" color="#64748B">
                          Deductible: {tmpl.deductible?.toLocaleString()} | Coverage keys: {Object.keys(tmpl.coverage).join(', ')}
                        </Typography>
                      </Box>
                    </Button>
                  ))}
                </Box>
              )}

              {/* Editable quick-create form */}
              {quickForm && (
                <Paper sx={{ p: 2, bgcolor: '#0F172A', borderRadius: 2, border: '1px solid #334155', mt: 1 }}>
                  <Typography variant="subtitle2" color="#94A3B8" fontWeight={600} sx={{ mb: 2 }}>
                    Customize & Create (defaults pre-filled for {countryName})
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={6} md={4}>
                      <TextField fullWidth label="Policy Number" value={quickForm.policy_number}
                        onChange={(e) => qf('policy_number', e.target.value)} size="small" />
                    </Grid>
                    <Grid item xs={6} md={4}>
                      <TextField fullWidth label="Holder Name" value={quickForm.holder_name}
                        onChange={(e) => qf('holder_name', e.target.value)} size="small"
                        placeholder="Your name" />
                    </Grid>
                    <Grid item xs={6} md={4}>
                      <TextField fullWidth select label="Type" value={quickForm.type} size="small"
                        onChange={(e) => qf('type', e.target.value)}>
                        <MenuItem value="auto">Auto</MenuItem>
                        <MenuItem value="homeowners">Homeowners</MenuItem>
                      </TextField>
                    </Grid>
                    <Grid item xs={6} md={3}>
                      <TextField fullWidth label="Start Date" type="date" value={quickForm.start_date} size="small"
                        InputLabelProps={{ shrink: true }} onChange={(e) => qf('start_date', e.target.value)} />
                    </Grid>
                    <Grid item xs={6} md={3}>
                      <TextField fullWidth label="End Date" type="date" value={quickForm.end_date} size="small"
                        InputLabelProps={{ shrink: true }} onChange={(e) => qf('end_date', e.target.value)} />
                    </Grid>
                    <Grid item xs={6} md={3}>
                      <TextField fullWidth label="Deductible" type="number" value={quickForm.deductible} size="small"
                        onChange={(e) => qf('deductible', Number(e.target.value))} />
                    </Grid>
                    <Grid item xs={6} md={3}>
                      <TextField fullWidth label="Premium/mo" type="number" value={quickForm.premium_monthly} size="small"
                        onChange={(e) => qf('premium_monthly', Number(e.target.value))} />
                    </Grid>
                  </Grid>

                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 2 }}>
                    <Typography variant="caption" color="#64748B" sx={{ flex: 1 }}>
                      Coverage: {Object.entries(quickForm.coverage || {}).map(([k, v]) => `${k}: ${v.toLocaleString()}`).join(' | ')}
                    </Typography>
                  </Box>

                  <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
                    <Button variant="contained" size="small" onClick={handleQuickSave} disabled={quickSaving || !quickForm.policy_number || !quickForm.holder_name}
                      startIcon={quickSaving ? <CircularProgress size={16} /> : <CheckCircle />}
                      sx={{ borderRadius: 2, background: 'linear-gradient(135deg, #10B981, #059669)' }}>
                      Create & Select
                    </Button>
                    <Button size="small" onClick={() => setQuickForm(null)} sx={{ color: '#94A3B8' }}>Cancel</Button>
                  </Box>
                </Paper>
              )}
            </Collapse>
          </>
        )}
      </Paper>

      {/* ── Step 2: Claim Details ─────────────────────────────────────────── */}
      <Paper component="form" onSubmit={handleSubmit} sx={{ p: 3, bgcolor: '#1E293B', borderRadius: 3, border: '1px solid #334155' }}>
        <Typography variant="subtitle1" fontWeight={700} color="#F1F5F9" sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
          <Send sx={{ color: '#3B82F6', fontSize: 20 }} /> Step 2: Claim Details
        </Typography>

        <Typography variant="subtitle2" color="#94A3B8" fontWeight={600} sx={{ mb: 2 }}>Incident</Typography>
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={12} md={6}>
            <TextField select fullWidth label="Incident Type" value={form.incident_type} onChange={handleChange('incident_type')}>
              {claimTypes.map((t) => <MenuItem key={t.value} value={t.value}>{t.label}</MenuItem>)}
            </TextField>
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField fullWidth label="Incident Date" type="date" value={form.incident_date} onChange={handleChange('incident_date')} InputLabelProps={{ shrink: true }} />
          </Grid>
          <Grid item xs={12}>
            <TextField fullWidth multiline rows={3} label="Incident Description *"
              value={form.incident_description} onChange={handleChange('incident_description')}
              placeholder={hints.description}
              helperText={`${form.incident_description.length}/500 characters (min 10)`}
              inputProps={{ maxLength: 500 }}
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField fullWidth label="Location" value={form.incident_location} onChange={handleChange('incident_location')}
              placeholder={hints.location} />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField fullWidth label="Police/FIR Report #" value={form.police_report_number} onChange={handleChange('police_report_number')}
              placeholder={hints.policeReport} />
          </Grid>
        </Grid>

        <Typography variant="subtitle2" color="#94A3B8" fontWeight={600} sx={{ mb: 2 }}>Damage Estimate</Typography>
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={12} md={3}>
            <TextField fullWidth label={`Estimated Amount (${currencySymbol}) *`} type="number" value={form.estimated_amount} onChange={handleChange('estimated_amount')}
              placeholder={hints.amount} inputProps={{ min: 0, step: 100 }} />
          </Grid>
          {isAuto && (
            <>
              <Grid item xs={12} md={3}>
                <TextField fullWidth label="Vehicle Year" type="number" value={form.vehicle_year} onChange={handleChange('vehicle_year')}
                  placeholder={hints.vehicleYear} inputProps={{ min: 1990, max: 2026 }} />
              </Grid>
              <Grid item xs={12} md={3}>
                <TextField fullWidth label="Vehicle Make" value={form.vehicle_make} onChange={handleChange('vehicle_make')}
                  placeholder={hints.vehicleMake} />
              </Grid>
              <Grid item xs={12} md={3}>
                <TextField fullWidth label="Vehicle Model" value={form.vehicle_model} onChange={handleChange('vehicle_model')}
                  placeholder={hints.vehicleModel} />
              </Grid>
            </>
          )}
        </Grid>

        <Alert severity="info" sx={{ mb: 3 }}>
          Your claim will be processed through our AI pipeline: intake validation, fraud check (CrewAI), damage assessment, policy compliance, settlement calculation, and quality evaluation. High-value or uncertain claims are escalated for human review.
        </Alert>

        <Button type="submit" variant="contained" size="large" disabled={!canSubmit || loading}
          startIcon={loading ? <CircularProgress size={20} /> : <Send />}
          sx={{ px: 4, py: 1.5, fontWeight: 600, borderRadius: 2, background: 'linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%)' }}
        >
          Submit Claim
        </Button>
      </Paper>
    </Box>
  );
}
