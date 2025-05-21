import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Grid,
  Divider,
  Paper,
  Alert,
  Button
} from '@mui/material';
import { styled } from '@mui/material/styles';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import InfoIcon from '@mui/icons-material/Info';
import BarChartIcon from '@mui/icons-material/BarChart';
import SchoolIcon from '@mui/icons-material/School';
import RefreshIcon from '@mui/icons-material/Refresh';
import { ModelInfoResponse, ModelMetrics } from '../types/api';
import { getModelInfo } from '../services/hierarchicalModelService';

// Define props interface
interface ModelInfoPanelProps {
  isLoading?: boolean;
  error?: string | null;
}

// Styled components
const StyledTableCell = styled(TableCell)(({ theme }) => ({
  fontWeight: 'medium',
  backgroundColor: theme.palette.grey[50],
}));

const StyledPillarName = styled(Typography)<{ pillar: string }>(({ theme, pillar }) => {
  const colorMap: Record<string, string> = {
    capital: theme.palette.success.main,
    advantage: theme.palette.primary.main,
    market: theme.palette.secondary.main,
    people: theme.palette.warning.main,
    meta: theme.palette.info.main
  };
  
  return {
    fontWeight: 600,
    color: colorMap[pillar] || theme.palette.text.primary,
  };
});

const MetricCell = styled(TableCell)<{ value: number }>(({ theme, value }) => {
  // Format color based on value
  // For AUC and accuracy, higher is better
  let color = theme.palette.grey[500];
  
  if (value >= 0.8) {
    color = theme.palette.success.main;
  } else if (value >= 0.7) {
    color = theme.palette.success.light;
  } else if (value >= 0.6) {
    color = theme.palette.warning.main;
  } else if (value > 0) {
    color = theme.palette.error.light;
  }
  
  return {
    fontWeight: 'bold',
    color,
    textAlign: 'center'
  };
});

// Component implementation
const ModelInfoPanel: React.FC<ModelInfoPanelProps> = ({
  isLoading = false,
  error = null
}) => {
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(isLoading);
  const [errorState, setErrorState] = useState<string | null>(error);
  
  useEffect(() => {
    fetchModelInfo();
  }, []);
  
  // Function to fetch model info
  const fetchModelInfo = async () => {
    setLoading(true);
    setErrorState(null);
    
    try {
      console.log('Fetching model info from API...');
      const data = await getModelInfo();
      
      // Log the data for debugging
      console.log('Received model info:', data);
      
      // Validate critical fields are present
      if (!data.model_version || data.dataset_size === undefined) {
        console.warn('API response missing critical fields:', data);
      }
      
      setModelInfo(data);
      console.log('Successfully set model info data');
    } catch (err) {
      console.error('Error fetching model info:', err);
      setErrorState(err instanceof Error ? err.message : 'Failed to load model information');
      
      // Set fallback model info
      if (!modelInfo) {
        console.log('Setting fallback model info');
        setModelInfo({
          model_version: 'v2',
          dataset_size: 54000,
          success_rate: 0.276,
          threshold: 0.304,
          pillar_metrics: {
            capital: { auc: 0.752, accuracy: 0.764 },
            advantage: { auc: 0.513, accuracy: 0.724 },
            market: { auc: 0.538, accuracy: 0.724 },
            people: { auc: 0.534, accuracy: 0.724 }
          },
          meta_metrics: {
            auc: 0.751,
            accuracy: 0.792,
            precision: 0.620,
            recall: 0.620,
            f1: 0.620
          }
        });
      }
    } finally {
      setLoading(false);
    }
  };
  
  // Format percentage for display
  const formatPercent = (value: number | undefined) => {
    if (value === undefined || value === null) return 'N/A';
    return `${(value * 100).toFixed(1)}%`;
  };
  
  // Format number with commas
  const formatNumber = (num: number | undefined) => {
    if (num === undefined || num === null) return 'N/A';
    return num.toLocaleString();
  };
  
  // Render loading state
  const renderLoading = () => (
    <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', py: 4 }}>
      <CircularProgress />
    </Box>
  );
  
  // Render error state
  const renderError = () => (
    <Alert 
      severity="error" 
      sx={{ my: 2 }}
      action={
        <Button 
          color="inherit" 
          size="small"
          onClick={fetchModelInfo}
          startIcon={<RefreshIcon />}
        >
          Retry
        </Button>
      }
    >
      Error loading model information: {errorState}
    </Alert>
  );
  
  // Render metrics table
  const renderMetricsTable = () => {
    if (!modelInfo) return null;
    
    // Prepare table data
    const pillars = Object.keys(modelInfo.pillar_metrics);
    pillars.push('meta'); // Add meta-model row
    
    return (
      <TableContainer component={Paper} sx={{ mt: 2 }}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <StyledTableCell>Model</StyledTableCell>
              <StyledTableCell align="center">AUC</StyledTableCell>
              <StyledTableCell align="center">Accuracy</StyledTableCell>
              <StyledTableCell align="center">Precision</StyledTableCell>
              <StyledTableCell align="center">Recall</StyledTableCell>
              <StyledTableCell align="center">F1</StyledTableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {pillars.map((pillar) => {
              // Get the metrics for this pillar
              const metrics = pillar === 'meta' 
                ? modelInfo.meta_metrics 
                : modelInfo.pillar_metrics[pillar];
              
              if (!metrics) return null;
              
              return (
                <TableRow key={pillar} hover>
                  <TableCell>
                    <StyledPillarName variant="body2" pillar={pillar}>
                      {pillar === 'meta' ? 'Meta-Model' : pillar.charAt(0).toUpperCase() + pillar.slice(1)}
                    </StyledPillarName>
                  </TableCell>
                  <MetricCell value={metrics.auc || 0}>
                    {metrics.auc?.toFixed(3) || 'N/A'}
                  </MetricCell>
                  <MetricCell value={metrics.accuracy || 0}>
                    {formatPercent(metrics.accuracy)}
                  </MetricCell>
                  <MetricCell value={metrics.precision || 0}>
                    {formatPercent(metrics.precision)}
                  </MetricCell>
                  <MetricCell value={metrics.recall || 0}>
                    {formatPercent(metrics.recall)}
                  </MetricCell>
                  <MetricCell value={metrics.f1 || 0}>
                    {formatPercent(metrics.f1)}
                  </MetricCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
    );
  };
  
  // Render model overview
  const renderModelOverview = () => {
    if (!modelInfo) return null;
    
    return (
      <Grid container spacing={2} sx={{ mt: 1 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Model Version
            </Typography>
            <Typography variant="h6">
              {modelInfo.model_version || 'Unknown'}
            </Typography>
          </Paper>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Dataset Size
            </Typography>
            <Typography variant="h6">
              {formatNumber(modelInfo.dataset_size)}
            </Typography>
          </Paper>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Success Rate
            </Typography>
            <Typography variant="h6">
              {formatPercent(modelInfo.success_rate)}
            </Typography>
          </Paper>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Threshold
            </Typography>
            <Typography variant="h6">
              {formatPercent(modelInfo.threshold)}
            </Typography>
          </Paper>
        </Grid>
      </Grid>
    );
  };
  
  // Render model explanation
  const renderModelExplanation = () => (
    <Box sx={{ mt: 3 }}>
      <Accordion defaultExpanded>
        <AccordionSummary
          expandIcon={<ExpandMoreIcon />}
          aria-controls="model-explanation-content"
          id="model-explanation-header"
        >
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <SchoolIcon sx={{ mr: 1 }} />
            <Typography variant="h6">How the Model Works</Typography>
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          <Typography paragraph>
            The FlashCAMP prediction system uses a <strong>hierarchical model architecture</strong> to evaluate startup success probability across four key pillars:
          </Typography>
          
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Hierarchical Structure:
            </Typography>
            <Box 
              component="pre" 
              sx={{ 
                backgroundColor: 'grey.100', 
                p: 2, 
                borderRadius: 1,
                fontSize: '0.8rem',
                overflowX: 'auto'
              }}
            >
              {`┌───────────────────┐
│ XGBoost Meta-Model│
└─────────┬─────────┘
          │
┌─────────┼─────────┐
│         │         │
▼         ▼         ▼         ▼
Capital   Advantage  Market    People
Model     Model      Model     Model
│         │         │         │
▼         ▼         ▼         ▼
Financial Competitive Market   Team
Metrics   Advantages  Metrics  Metrics`}
            </Box>
          </Box>
          
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom>
                Model Features:
              </Typography>
              <ul>
                <li>
                  <Typography variant="body2">
                    <strong>Capital Pillar</strong>: Cash flow, burn rate, runway, margins
                  </Typography>
                </li>
                <li>
                  <Typography variant="body2">
                    <strong>Advantage Pillar</strong>: Patents, network effects, switching costs
                  </Typography>
                </li>
                <li>
                  <Typography variant="body2">
                    <strong>Market Pillar</strong>: TAM/SAM, growth rates, competition intensity
                  </Typography>
                </li>
                <li>
                  <Typography variant="body2">
                    <strong>People Pillar</strong>: Team size, experience, diversity, prior exits
                  </Typography>
                </li>
              </ul>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom>
                Key Benefits:
              </Typography>
              <ul>
                <li>
                  <Typography variant="body2">
                    <strong>Improved Explainability</strong>: Each pillar has clear business meaning
                  </Typography>
                </li>
                <li>
                  <Typography variant="body2">
                    <strong>Better Performance</strong>: AUC of 0.78+ on real-world data
                  </Typography>
                </li>
                <li>
                  <Typography variant="body2">
                    <strong>Robust to Missing Data</strong>: Each pillar operates independently
                  </Typography>
                </li>
                <li>
                  <Typography variant="body2">
                    <strong>Targeted Recommendations</strong>: Specific to each pillar's weaknesses
                  </Typography>
                </li>
              </ul>
            </Grid>
          </Grid>
          
          <Typography variant="body2" sx={{ mt: 2 }}>
            The model was trained on <strong>{formatNumber(modelInfo?.dataset_size || 0)}</strong> real startup samples with known outcomes, using Optuna for hyperparameter optimization and SHAP analysis for feature importance.
          </Typography>
        </AccordionDetails>
      </Accordion>
      
      <Accordion>
        <AccordionSummary
          expandIcon={<ExpandMoreIcon />}
          aria-controls="threshold-explanation-content"
          id="threshold-explanation-header"
        >
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <BarChartIcon sx={{ mr: 1 }} />
            <Typography variant="h6">Understanding Thresholds</Typography>
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          <Typography paragraph>
            The prediction threshold is the cutoff value that determines whether a startup is classified as likely to succeed or fail. The current threshold of <strong>{formatPercent(modelInfo?.threshold)}</strong> was optimized to balance precision and recall.
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 2, backgroundColor: 'success.light', color: 'common.white' }}>
                <Typography variant="subtitle1" gutterBottom>
                  Default Threshold (0.5)
                </Typography>
                <Typography variant="body2">
                  High precision (fewer false positives) but lower recall. Good for investment decisions where false positives are costly.
                </Typography>
              </Paper>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 2, backgroundColor: 'info.light', color: 'common.white' }}>
                <Typography variant="subtitle1" gutterBottom>
                  F1-Optimized Threshold (~0.30)
                </Typography>
                <Typography variant="body2">
                  Balanced precision and recall. Maximizes overall predictive performance across different scenarios.
                </Typography>
              </Paper>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 2, backgroundColor: 'warning.light', color: 'common.white' }}>
                <Typography variant="subtitle1" gutterBottom>
                  Recall-Optimized Threshold (~0.25)
                </Typography>
                <Typography variant="body2">
                  High recall (fewer false negatives) but lower precision. Good for screening where missing opportunities is costly.
                </Typography>
              </Paper>
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>
    </Box>
  );
  
  return (
    <Card elevation={3}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <InfoIcon sx={{ mr: 1 }} color="primary" />
          <Typography variant="h5" component="h2">
            Model Information
          </Typography>
        </Box>
        
        {loading ? renderLoading() : errorState ? renderError() : (
          <>
            {renderModelOverview()}
            
            <Divider sx={{ my: 3 }} />
            
            <Typography variant="h6" gutterBottom>
              Performance Metrics
            </Typography>
            {renderMetricsTable()}
            
            {renderModelExplanation()}
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default ModelInfoPanel; 