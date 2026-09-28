import React from 'react';
import { Box, Grid, Typography } from '@mui/material';
import { HubOutlined } from '@mui/icons-material';
import RepoCard from './RepoCard';

// Grid of repo lanes. Purely presentational so both the live console and the
// read-only history replay can render it from the same derived repo list.
export default function FleetBoard({ repos = [], readOnly = false, emptyHint }) {
  if (repos.length === 0) {
    return (
      <Box
        sx={{
          minHeight: 260,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          textAlign: 'center',
          color: 'text.secondary',
          gap: 1,
        }}
      >
        <HubOutlined sx={{ fontSize: 48, opacity: 0.4 }} />
        <Typography variant="body1">No repositories in flight</Typography>
        <Typography variant="body2" sx={{ maxWidth: 380 }}>
          {emptyHint ||
            'Pick a migration job above and run it. Each repository gets its own lane and updates live as the agents work.'}
        </Typography>
      </Box>
    );
  }

  return (
    <Grid container spacing={2} alignItems="stretch">
      {repos.map((repo) => (
        <Grid item xs={12} sm={6} lg={4} key={repo.repo_id}>
          <RepoCard repo={repo} readOnly={readOnly} />
        </Grid>
      ))}
    </Grid>
  );
}
