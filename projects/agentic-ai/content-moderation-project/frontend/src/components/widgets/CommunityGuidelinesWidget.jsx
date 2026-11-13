import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Divider
} from '@mui/material';

const CommunityGuidelinesWidget = () => {
  return (
    <Card elevation={2}>
      <CardContent>
        <Typography variant="h5" component="h2" gutterBottom sx={{
          fontWeight: "bold"
        }}>
          Community Guidelines
        </Typography>
        <Divider sx={{ mb: 3 }} />

        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle1" gutterBottom sx={{
            fontWeight: "bold"
          }}>
            Be Respectful:
          </Typography>
          <Typography variant="body2" sx={{
            color: "text.secondary"
          }}>
            Treat others with kindness and respect.
          </Typography>
        </Box>

        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle1" gutterBottom sx={{
            fontWeight: "bold"
          }}>
            No Hate Speech:
          </Typography>
          <Typography variant="body2" sx={{
            color: "text.secondary"
          }}>
            Discrimination based on race, gender, religion, etc. is not allowed.
          </Typography>
        </Box>

        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle1" gutterBottom sx={{
            fontWeight: "bold"
          }}>
            No Harassment:
          </Typography>
          <Typography variant="body2" sx={{
            color: "text.secondary"
          }}>
            Bullying, threats, or targeted harassment will result in removal.
          </Typography>
        </Box>

        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle1" gutterBottom sx={{
            fontWeight: "bold"
          }}>
            Stay On Topic:
          </Typography>
          <Typography variant="body2" sx={{
            color: "text.secondary"
          }}>
            Keep discussions relevant and constructive.
          </Typography>
        </Box>

        <Typography
          variant="body2"
          sx={{
            color: "text.secondary",
            mt: 2,
            fontStyle: 'italic'
          }}>
          Violations may result in content removal, warnings, or account suspension.
        </Typography>
      </CardContent>
    </Card>
  );
};

export default CommunityGuidelinesWidget;
