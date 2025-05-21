import React, { useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Grid,
  Button,
  Tooltip,
  CircularProgress,
  Alert
} from '@mui/material';
import { styled } from '@mui/material/styles';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ArrowRightIcon from '@mui/icons-material/ArrowRight';
import TipsAndUpdatesIcon from '@mui/icons-material/TipsAndUpdates';
import PriorityHighIcon from '@mui/icons-material/PriorityHigh';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import LocalFireDepartmentIcon from '@mui/icons-material/LocalFireDepartment';
import { RecommendationsResponse, RecommendationItem } from '../types/api';

// Define props interface
interface RecommendationsPanelProps {
  recommendations: RecommendationsResponse | null;
  isLoading?: boolean;
  error?: string | null;
  onApplyRecommendation?: (pillar: string, metric: string) => void;
}

// Styled components
const PillarCard = styled(Card)(({ theme }) => ({
  marginBottom: theme.spacing(2),
  borderLeft: `4px solid ${theme.palette.primary.main}`,
}));

const PillarTitle = styled(Typography)(({ theme }) => ({
  fontWeight: 600,
  fontSize: '1.1rem',
  padding: theme.spacing(1.5),
  backgroundColor: theme.palette.grey[50],
  borderBottom: `1px solid ${theme.palette.divider}`,
}));

const ImpactChip = styled(Chip)<{ impact: string }>(({ theme, impact }) => {
  // Define colors for different impact levels
  const impactColors = {
    high: {
      background: theme.palette.error.light,
      color: theme.palette.error.contrastText,
    },
    medium: {
      background: theme.palette.warning.light,
      color: theme.palette.warning.contrastText,
    },
    low: {
      background: theme.palette.info.light,
      color: theme.palette.info.contrastText,
    }
  };
  
  // Default to medium if impact not specified
  const colorSet = impactColors[impact as keyof typeof impactColors] || impactColors.medium;
  
  return {
    backgroundColor: colorSet.background,
    color: colorSet.color,
    fontWeight: 600,
    fontSize: '0.7rem',
  };
});

// Component implementation
const RecommendationsPanel: React.FC<RecommendationsPanelProps> = ({
  recommendations,
  isLoading = false,
  error = null,
  onApplyRecommendation
}) => {
  const [expandedPillar, setExpandedPillar] = React.useState<string>('');
  
  // Handle panel expansion
  const handleChange = (pillar: string) => (event: React.SyntheticEvent, isExpanded: boolean) => {
    setExpandedPillar(isExpanded ? pillar : '');
  };
  
  // Create a sorted list of pillars by recommendation count
  const sortedPillars = useMemo(() => {
    if (!recommendations) return [];
    
    return Object.entries(recommendations)
      .filter(([_, recs]) => recs && recs.length > 0) // Only include pillars with recommendations
      .sort((a, b) => b[1].length - a[1].length); // Sort by number of recommendations
  }, [recommendations]);
  
  // Format pillar name for display
  const formatPillarName = (name: string) => {
    return name.charAt(0).toUpperCase() + name.slice(1);
  };
  
  // Get icon for impact level
  const getImpactIcon = (impact: string) => {
    switch (impact.toLowerCase()) {
      case 'high':
        return <LocalFireDepartmentIcon fontSize="small" />;
      case 'medium':
        return <PriorityHighIcon fontSize="small" />;
      default:
        return <TipsAndUpdatesIcon fontSize="small" />;
    }
  };
  
  // Get color for pillar
  const getPillarColor = (pillar: string) => {
    const colorMap: Record<string, string> = {
      capital: '#2E7D32', // Green
      advantage: '#1976D2', // Blue
      market: '#9C27B0', // Purple
      people: '#ED6C02', // Orange
    };
    
    return colorMap[pillar] || '#666666';
  };
  
  // Render loading state
  if (isLoading) {
    return (
      <Card elevation={3}>
        <CardContent>
          <Box sx={{ 
            display: 'flex', 
            flexDirection: 'column', 
            alignItems: 'center', 
            justifyContent: 'center',
            minHeight: 200,
            gap: 2
          }}>
            <CircularProgress size={40} />
            <Typography variant="body1" color="text.secondary">
              Loading recommendations...
            </Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }
  
  // Render error state
  if (error) {
    return (
      <Card elevation={3}>
        <CardContent>
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
          <Typography variant="body2" color="text.secondary">
            Please try again later or contact support if the problem persists.
          </Typography>
        </CardContent>
      </Card>
    );
  }
  
  // Render empty state if no recommendations
  if (!recommendations || sortedPillars.length === 0) {
    return (
      <Card elevation={3}>
        <CardContent>
          <Box sx={{ textAlign: 'center', py: 3 }}>
            <TipsAndUpdatesIcon sx={{ fontSize: 40, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              No Recommendations Available
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Your startup is performing well across all pillars or insufficient data was provided.
            </Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <Card elevation={3}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <TipsAndUpdatesIcon sx={{ mr: 1 }} color="primary" />
          <Typography variant="h5" component="h2">
            Recommendations
          </Typography>
        </Box>
        
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Based on your startup metrics, we've identified the following recommendations 
          to improve your success probability.
        </Typography>
        
        {sortedPillars.map(([pillar, pillarRecs]) => (
          <Accordion 
            key={pillar}
            defaultExpanded={pillar === sortedPillars[0][0]}
            sx={{ 
              mb: 1.5,
              borderLeft: `4px solid ${getPillarColor(pillar)}`,
              '&.Mui-expanded': {
                margin: '0 0 12px 0',
              }
            }}
          >
            <AccordionSummary
              expandIcon={<ExpandMoreIcon />}
              aria-controls={`${pillar}-content`}
              id={`${pillar}-header`}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', width: '100%', justifyContent: 'space-between' }}>
                <Typography variant="subtitle1" fontWeight="medium">
                  {formatPillarName(pillar)} Pillar
                </Typography>
                <Chip 
                  label={`${pillarRecs.length} recommendations`} 
                  size="small" 
                  color="primary" 
                  variant="outlined"
                />
              </Box>
            </AccordionSummary>
            
            <AccordionDetails>
              <List disablePadding>
                {pillarRecs.map((rec: RecommendationItem, index: number) => (
                  <React.Fragment key={`${pillar}-rec-${index}`}>
                    {index > 0 && <Divider component="li" />}
                    <ListItem alignItems="flex-start">
                      <ArrowRightIcon 
                        sx={{ 
                          mt: 0.5, 
                          mr: 1,
                          color: 'primary.main'
                        }} 
                      />
                      <ListItemText
                        primary={
                          <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
                            <Typography component="span" variant="body1" fontWeight="medium">
                              {rec.recommendation}
                            </Typography>
                            <Tooltip title={`${rec.impact.toUpperCase()} impact`}>
                              <Box sx={{ ml: 1, mt: 0.5 }}>
                                <ImpactChip 
                                  icon={getImpactIcon(rec.impact)}
                                  label={rec.impact} 
                                  size="small" 
                                  impact={rec.impact} 
                                />
                              </Box>
                            </Tooltip>
                          </Box>
                        }
                        secondary={
                          <Typography
                            component="span"
                            variant="body2"
                            color="text.secondary"
                          >
                            Metric: {rec.metric}
                          </Typography>
                        }
                      />
                    </ListItem>
                  </React.Fragment>
                ))}
              </List>
            </AccordionDetails>
          </Accordion>
        ))}
      </CardContent>
    </Card>
  );
};

export default RecommendationsPanel; 