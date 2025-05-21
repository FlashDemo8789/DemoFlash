import React, { useState } from 'react';
import { 
  Box, 
  Card, 
  CardContent, 
  Typography, 
  Grid, 
  LinearProgress, 
  Chip, 
  Button,
  Paper,
  Switch,
  FormControlLabel,
  Divider,
  Tooltip,
  IconButton
} from '@mui/material';
import { styled } from '@mui/material/styles';
import InfoIcon from '@mui/icons-material/Info';
import BarChartIcon from '@mui/icons-material/BarChart';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import { HierarchicalPredictionResponse, PillarScores } from '../types/api';

// Define props interface
interface HierarchicalModelResultsProps {
  prediction: HierarchicalPredictionResponse;
  showVisualization?: boolean;
  onViewVisualization?: () => void;
}

// Styled components
const ScoreBar = styled(LinearProgress)<{ value: number }>(({ theme, value }) => ({
  height: 16,
  borderRadius: 5,
  [`&.MuiLinearProgress-colorPrimary`]: {
    backgroundColor: theme.palette.grey[300],
  },
  [`& .MuiLinearProgress-bar`]: {
    borderRadius: 5,
    backgroundColor: value >= 0.7 
      ? theme.palette.success.main 
      : value >= 0.5 
        ? theme.palette.warning.main 
        : theme.palette.error.main,
  },
}));

const PillarTitle = styled(Typography)({
  fontWeight: 600,
  fontSize: '1rem',
  marginBottom: '4px',
});

const ThresholdIndicator = styled('div')(({ theme }) => ({
  position: 'absolute',
  width: '2px',
  height: '24px',
  backgroundColor: theme.palette.info.main,
  top: -4,
  '&::after': {
    content: '""',
    position: 'absolute',
    bottom: '-6px',
    left: '-4px',
    width: 0,
    height: 0,
    borderLeft: '5px solid transparent',
    borderRight: '5px solid transparent',
    borderTop: `6px solid ${theme.palette.info.main}`,
  }
}));

// Component implementation
const HierarchicalModelResults: React.FC<HierarchicalModelResultsProps> = ({
  prediction,
  showVisualization = true,
  onViewVisualization
}) => {
  const [showDetailed, setShowDetailed] = useState(false);
  
  // Extract data from prediction
  const { pillar_scores, final_score, prediction: result, confidence, threshold } = prediction;
  
  // Format percentage for display
  const formatPercent = (value: number) => `${Math.round(value * 100)}%`;
  
  // Get color based on score
  const getScoreColor = (score: number) => {
    if (score >= 0.7) return 'success.main';
    if (score >= 0.5) return 'warning.main';
    return 'error.main';
  };
  
  // Get icon based on prediction
  const getPredictionIcon = () => {
    return result === 'pass' 
      ? <CheckCircleIcon fontSize="large" color="success" /> 
      : <CancelIcon fontSize="large" color="error" />;
  };
  
  // Get pillar title with first letter capitalized
  const formatPillarName = (pillar: string) => {
    return pillar.charAt(0).toUpperCase() + pillar.slice(1);
  };
  
  // Render pillar score bar
  const renderPillarScore = (pillar: string, score: number) => {
    const leftPosition = `${threshold * 100}%`;
    
    return (
      <Box key={pillar} sx={{ mb: 2, position: 'relative' }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
          <PillarTitle>{formatPillarName(pillar)}</PillarTitle>
          <Typography color={getScoreColor(score)} fontWeight="bold">
            {formatPercent(score)}
          </Typography>
        </Box>
        <Box sx={{ position: 'relative' }}>
          <ScoreBar 
            variant="determinate" 
            value={score * 100} 
            value-data={score}
          />
          <Box sx={{ position: 'absolute', left: leftPosition, top: 0 }}>
            <ThresholdIndicator />
          </Box>
        </Box>
      </Box>
    );
  };
  
  // Render detailed view with additional metrics
  const renderDetailedView = () => {
    return (
      <Box sx={{ mt: 3 }}>
        <Divider sx={{ mb: 2 }} />
        <Typography variant="h6" gutterBottom>
          Detailed Metrics
        </Typography>
        
        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                Threshold
              </Typography>
              <Typography variant="h6">{formatPercent(threshold)}</Typography>
            </Paper>
          </Grid>
          <Grid item xs={6}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                Confidence
              </Typography>
              <Typography variant="h6">{formatPercent(confidence)}</Typography>
            </Paper>
          </Grid>
        </Grid>
        
        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Score Interpretation:
          </Typography>
          <Typography variant="body2" paragraph>
            <Box component="span" sx={{ color: 'success.main', fontWeight: 'bold' }}>
              Green (≥70%)
            </Box>
            : Strong performance in this pillar
          </Typography>
          <Typography variant="body2" paragraph>
            <Box component="span" sx={{ color: 'warning.main', fontWeight: 'bold' }}>
              Yellow (50-69%)
            </Box>
            : Moderate performance, potential for improvement
          </Typography>
          <Typography variant="body2" paragraph>
            <Box component="span" sx={{ color: 'error.main', fontWeight: 'bold' }}>
              Red (≤49%)
            </Box>
            : Weak performance, significant improvement needed
          </Typography>
        </Box>
      </Box>
    );
  };
  
  return (
    <Card elevation={3}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h5" component="h2">
            Startup Success Prediction
          </Typography>
          <Box>
            {showVisualization && (
              <Tooltip title="View detailed visualization">
                <IconButton 
                  color="primary" 
                  onClick={onViewVisualization}
                  sx={{ mr: 1 }}
                >
                  <BarChartIcon />
                </IconButton>
              </Tooltip>
            )}
            <Tooltip title="Learn more about the model">
              <IconButton color="info">
                <InfoIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
        
        <Box sx={{ mb: 3, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mr: 2 }}>
            {getPredictionIcon()}
          </Box>
          <Box>
            <Typography variant="h4" component="div" sx={{ fontWeight: 'bold' }}>
              {result.toUpperCase()}
            </Typography>
            <Chip 
              label={`${formatPercent(confidence)} confident`} 
              color={confidence > 0.8 ? "success" : "primary"}
              size="small"
            />
          </Box>
        </Box>
        
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle1" gutterBottom>
            Final Score: <Box component="span" sx={{ fontWeight: 'bold', color: getScoreColor(final_score) }}>
              {formatPercent(final_score)}
            </Box>
          </Typography>
        </Box>
        
        <Typography variant="subtitle1" gutterBottom>
          Pillar Scores:
        </Typography>
        
        <Box sx={{ mt: 2 }}>
          {Object.entries(pillar_scores).map(([pillar, score]) => 
            renderPillarScore(pillar, score)
          )}
        </Box>
        
        <Box sx={{ mt: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <FormControlLabel
            control={
              <Switch 
                checked={showDetailed}
                onChange={(e) => setShowDetailed(e.target.checked)}
                color="primary"
              />
            }
            label="Show detailed view"
          />
          
          {onViewVisualization && (
            <Button 
              variant="outlined" 
              startIcon={<BarChartIcon />}
              onClick={onViewVisualization}
            >
              View Visualization
            </Button>
          )}
        </Box>
        
        {showDetailed && renderDetailedView()}
      </CardContent>
    </Card>
  );
};

export default HierarchicalModelResults; 