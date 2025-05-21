import React, { useState } from 'react';
import {
  Box,
  Card,
  Paper,
  Typography,
  Button,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  alpha,
  Divider,
  IconButton,
  Tooltip
} from '@mui/material';
import { styled } from '@mui/material/styles';
import CloseIcon from '@mui/icons-material/Close';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import BarChartIcon from '@mui/icons-material/BarChart';
import RecommendIcon from '@mui/icons-material/Recommend';
import LightbulbIcon from '@mui/icons-material/Lightbulb';
import AssessmentIcon from '@mui/icons-material/Assessment';

// Define props interface
interface HierarchicalModelOnboardingProps {
  onClose: () => void;
  onComplete: () => void;
}

// Styled components
const OnboardingCard = styled(Card)(({ theme }) => ({
  position: 'relative',
  overflow: 'visible',
  borderRadius: 16,
  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.12)',
  border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
}));

const OnboardingHeader = styled(Box)(({ theme }) => ({
  padding: theme.spacing(3),
  background: `linear-gradient(135deg, ${alpha(theme.palette.primary.dark, 0.8)} 0%, ${alpha(theme.palette.primary.main, 0.9)} 100%)`,
  color: theme.palette.common.white,
  borderTopLeftRadius: 16,
  borderTopRightRadius: 16,
}));

const StepImage = styled(Box)(({ theme }) => ({
  backgroundColor: alpha(theme.palette.primary.main, 0.1),
  borderRadius: 8,
  padding: 0,
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  marginBottom: theme.spacing(2),
  height: 160,
}));

// Define onboarding steps
const onboardingSteps = [
  {
    label: 'Introducing the Hierarchical Model',
    description: 'Our new Hierarchical Model analyzes your startup across four key pillars: Capital, Advantage, Market, and People. Each pillar contributes to your overall success prediction, providing deeper insights and better explainability.',
    icon: <BarChartIcon fontSize="large" />,
    color: '#3B82F6',
  },
  {
    label: 'Understand Your Pillar Scores',
    description: 'Each pillar has its own prediction score and confidence level. Strong pillars improve your overall prediction, while weak pillars highlight areas that need improvement.',
    icon: <AssessmentIcon fontSize="large" />,
    color: '#10B981',
  },
  {
    label: 'Get Actionable Recommendations',
    description: 'The model provides tailored recommendations for each pillar. Focus on high-impact recommendations to improve your startup\'s success probability most effectively.',
    icon: <LightbulbIcon fontSize="large" />,
    color: '#F59E0B',
  },
  {
    label: 'Visualize Prediction Factors',
    description: 'Interactive visualizations help you understand which metrics have the biggest influence on your prediction. Use this information to make data-driven decisions.',
    icon: <RecommendIcon fontSize="large" />,
    color: '#8B5CF6',
  }
];

// Main component implementation
const HierarchicalModelOnboarding: React.FC<HierarchicalModelOnboardingProps> = ({
  onClose,
  onComplete
}) => {
  const [activeStep, setActiveStep] = useState(0);
  
  // Handle step navigation
  const handleNext = () => {
    setActiveStep((prevStep) => prevStep + 1);
  };
  
  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };
  
  const handleComplete = () => {
    onComplete();
  };
  
  return (
    <OnboardingCard>
      <OnboardingHeader>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h5" component="h2" fontWeight="600">
            Hierarchical Model Guide
          </Typography>
          <Tooltip title="Close guide">
            <IconButton onClick={onClose} size="small" sx={{ color: 'white' }}>
              <CloseIcon />
            </IconButton>
          </Tooltip>
        </Box>
        <Typography variant="body2" sx={{ mt: 1, opacity: 0.9 }}>
          Learn how to use our new advanced prediction model
        </Typography>
      </OnboardingHeader>
      
      <Box sx={{ p: 3 }}>
        <Stepper activeStep={activeStep} orientation="vertical">
          {onboardingSteps.map((step, index) => (
            <Step key={step.label}>
              <StepLabel>
                <Typography variant="subtitle1" fontWeight="600">
                  {step.label}
                </Typography>
              </StepLabel>
              <StepContent>
                <StepImage>
                  <Box sx={{ 
                    p: 3, 
                    borderRadius: '50%', 
                    bgcolor: alpha(step.color, 0.15),
                    color: step.color,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: 80,
                    height: 80,
                    fontSize: 40
                  }}>
                    {step.icon}
                  </Box>
                </StepImage>
                
                <Typography variant="body1" sx={{ mb: 2 }}>
                  {step.description}
                </Typography>
                
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 2 }}>
                  <Button
                    disabled={index === 0}
                    onClick={handleBack}
                    startIcon={<ChevronLeftIcon />}
                    sx={{ mr: 1 }}
                  >
                    Back
                  </Button>
                  <Box>
                    {index === onboardingSteps.length - 1 ? (
                      <Button
                        variant="contained"
                        onClick={handleComplete}
                        endIcon={<CheckCircleIcon />}
                        sx={{ borderRadius: 2 }}
                      >
                        Get Started
                      </Button>
                    ) : (
                      <Button
                        variant="outlined"
                        onClick={handleNext}
                        endIcon={<ChevronRightIcon />}
                        sx={{ borderRadius: 2 }}
                      >
                        Continue
                      </Button>
                    )}
                  </Box>
                </Box>
              </StepContent>
            </Step>
          ))}
        </Stepper>
      </Box>
      
      <Divider />
      
      <Box sx={{ p: 2, display: 'flex', justifyContent: 'flex-end' }}>
        <Button 
          size="small" 
          onClick={onClose}
          sx={{ mr: 1, textTransform: 'none' }}
        >
          Skip for now
        </Button>
        <Button 
          size="small" 
          variant="text" 
          color="primary"
          onClick={handleComplete}
          sx={{ textTransform: 'none' }}
        >
          Don't show again
        </Button>
      </Box>
    </OnboardingCard>
  );
};

export default HierarchicalModelOnboarding; 