import React, { useEffect, useState } from 'react';
import {
  Box, Paper, Typography, Grid, CircularProgress, Chip, Table, TableBody,
  TableCell, TableContainer, TableHead, TableRow, LinearProgress,
} from '@mui/material';
import { Analytics, TrendingUp, AttachMoney, Security, Speed, Gavel } from '@mui/icons-material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend } from 'recharts';
import { analyticsAPI } from '../../services/api';
import { fmt } from '../../services/currency';

const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];

export default function AnalyticsDashboard() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await analyticsAPI.getMetrics();
        setMetrics(res.data);
      } catch {
        // Use demo data if API not connected
        setMetrics({
          total_claims: 127,
          avg_processing_time_sec: 32,
          total_cost_usd: 1.24,
          avg_fraud_score: 0.23,
          hitl_rate: 0.18,
          approval_rate: 0.72,
          avg_settlement_usd: 8450,
          avg_eval_score: 0.84,
          claims_by_type: [
            { type: 'Auto Collision', count: 52 },
            { type: 'Auto Theft', count: 18 },
            { type: 'Property Fire', count: 22 },
            { type: 'Property Water', count: 15 },
            { type: 'Medical', count: 12 },
            { type: 'Liability', count: 8 },
          ],
          decisions: [
            { decision: 'Approved', count: 58 },
            { decision: 'Partial', count: 22 },
            { decision: 'Denied', count: 18 },
            { decision: 'HITL', count: 15 },
            { decision: 'Auto Reject', count: 4 },
          ],
          pipeline_paths: [
            { path: 'A: Normal', count: 72, avg_cost: 0.008, avg_agents: 7 },
            { path: 'B: HITL', count: 23, avg_cost: 0.014, avg_agents: 9 },
            { path: 'C: Reject', count: 4, avg_cost: 0.004, avg_agents: 2 },
            { path: 'D: Intake', count: 12, avg_cost: 0.001, avg_agents: 2 },
            { path: 'E: Fast', count: 16, avg_cost: 0.003, avg_agents: 3 },
          ],
        });
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) return <Box sx={{ textAlign: 'center', py: 10 }}><CircularProgress /></Box>;
  const m = metrics || {};

  return (
    <Box>
      <Typography variant="h5" fontWeight={700} color="#F1F5F9" gutterBottom>
        <Analytics sx={{ mr: 1, verticalAlign: 'middle', color: '#3B82F6' }} />
        Pipeline Analytics
      </Typography>
      <Typography variant="body2" color="#94A3B8" sx={{ mb: 3 }}>Performance metrics and cost breakdown</Typography>

      {/* Key Metrics */}
      <Grid container spacing={2} sx={{ mb: 4 }}>
        {[
          { label: 'Total Claims', value: m.total_claims, icon: <Speed />, color: '#3B82F6' },
          { label: 'Approval Rate', value: `${(m.approval_rate * 100).toFixed(0)}%`, icon: <TrendingUp />, color: '#10B981' },
          { label: 'HITL (Human-In-The-Loop) Rate', value: `${(m.hitl_rate * 100).toFixed(0)}%`, icon: <Security />, color: '#F59E0B' },
          { label: 'Avg Processing', value: `${m.avg_processing_time_sec}s`, icon: <Speed />, color: '#8B5CF6' },
          { label: 'Total Cost', value: fmt(m.total_cost_usd), icon: <AttachMoney />, color: '#EF4444' },
          { label: 'Avg Eval Score', value: `${(m.avg_eval_score * 100).toFixed(0)}%`, icon: <Gavel />, color: '#0EA5E9' },
        ].map((stat) => (
          <Grid item xs={6} md={2} key={stat.label}>
            <Paper sx={{ p: 2, bgcolor: '#1E293B', borderRadius: 2, border: '1px solid #334155' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                <Box sx={{ color: stat.color }}>{stat.icon}</Box>
                <Typography variant="caption" color="#94A3B8">{stat.label}</Typography>
              </Box>
              <Typography variant="h5" fontWeight={700} sx={{ color: stat.color }}>{stat.value}</Typography>
            </Paper>
          </Grid>
        ))}
      </Grid>

      <Grid container spacing={3}>
        {/* Claims by Type - Bar Chart */}
        <Grid item xs={12} md={7}>
          <Paper sx={{ p: 3, bgcolor: '#1E293B', borderRadius: 3, border: '1px solid #334155' }}>
            <Typography variant="subtitle1" fontWeight={600} color="#F1F5F9" gutterBottom>Claims by Type</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={m.claims_by_type || []}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="type" tick={{ fill: '#94A3B8', fontSize: 11 }} />
                <YAxis tick={{ fill: '#94A3B8' }} />
                <Tooltip contentStyle={{ background: '#0F172A', border: '1px solid #334155', borderRadius: 8, color: '#F1F5F9' }} />
                <Bar dataKey="count" fill="#3B82F6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Decisions - Pie Chart */}
        <Grid item xs={12} md={5}>
          <Paper sx={{ p: 3, bgcolor: '#1E293B', borderRadius: 3, border: '1px solid #334155' }}>
            <Typography variant="subtitle1" fontWeight={600} color="#F1F5F9" gutterBottom>Decision Distribution</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie data={m.decisions || []} dataKey="count" nameKey="decision" cx="50%" cy="50%" outerRadius={100} label={({ decision, percent }) => `${decision} ${(percent * 100).toFixed(0)}%`}>
                  {(m.decisions || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip contentStyle={{ background: '#0F172A', border: '1px solid #334155', borderRadius: 8, color: '#F1F5F9' }} />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Pipeline Paths Table */}
        <Grid item xs={12}>
          <Paper sx={{ bgcolor: '#1E293B', borderRadius: 3, border: '1px solid #334155' }}>
            <Typography variant="subtitle1" fontWeight={600} color="#F1F5F9" sx={{ p: 2, pb: 1 }}>Pipeline Path Performance</Typography>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Path</TableCell>
                    <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Claims</TableCell>
                    <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Distribution</TableCell>
                    <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Avg Cost</TableCell>
                    <TableCell sx={{ color: '#64748B', fontWeight: 600 }}>Avg Agents</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {(m.pipeline_paths || []).map((p, i) => (
                    <TableRow key={p.path}>
                      <TableCell sx={{ color: '#F1F5F9', fontWeight: 600 }}>{p.path}</TableCell>
                      <TableCell sx={{ color: '#94A3B8' }}>{p.count}</TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <LinearProgress variant="determinate" value={(p.count / m.total_claims) * 100}
                            sx={{ width: 100, height: 8, borderRadius: 4, bgcolor: '#334155', '& .MuiLinearProgress-bar': { bgcolor: COLORS[i], borderRadius: 4 } }} />
                          <Typography variant="caption" color="#94A3B8">{((p.count / m.total_claims) * 100).toFixed(0)}%</Typography>
                        </Box>
                      </TableCell>
                      <TableCell sx={{ color: '#10B981', fontWeight: 600 }}>{fmt(p.avg_cost)}</TableCell>
                      <TableCell sx={{ color: '#94A3B8' }}>{p.avg_agents}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}
