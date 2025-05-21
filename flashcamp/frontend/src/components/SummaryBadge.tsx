import React, { useState, useEffect } from "react";
import { 
  Box, 
  Typography, 
  CircularProgress, 
  Paper,
  alpha,
  useTheme,
  useMediaQuery,
  LinearProgress
} from "@mui/material";
import { 
  Rocket as RocketIcon,
  TrendingUp as TrendingUpIcon,
  ArrowUpward as ArrowUpIcon,
  CheckCircle as CheckIcon,
  ErrorOutline as WarningIcon
} from "@mui/icons-material";
import { motion } from "framer-motion";

interface SummaryBadgeProps {
  prob: number;
}

const SummaryBadge: React.FC<SummaryBadgeProps> = ({ prob }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const [animatedProb, setAnimatedProb] = useState(0);
  const [category, setCategory] = useState<string>("");
  const [color, setColor] = useState<string>("#3B82F6");
  const [icon, setIcon] = useState<React.ReactNode>(<RocketIcon />);
  
  // Calculate percentage
  const percentage = prob * 100;
  
  // Determine category, color and icon based on probability
  useEffect(() => {
    if (percentage >= 80) {
      setCategory("Exceptional");
      setColor("#10B981");
      setIcon(<CheckIcon />);
    } else if (percentage >= 65) {
      setCategory("Strong");
      setColor("#3B82F6");
      setIcon(<RocketIcon />); 
    } else if (percentage >= 50) {
      setCategory("Promising");
      setColor("#8B5CF6");
      setIcon(<ArrowUpIcon />);
    } else if (percentage >= 35) {
      setCategory("Developing");
      setColor("#F59E0B");
      setIcon(<TrendingUpIcon />);
    } else {
      setCategory("Challenging");
      setColor("#EF4444");
      setIcon(<WarningIcon />);
    }
    
    // Animate the probability counting up
    const timer = setTimeout(() => {
      setAnimatedProb(percentage);
    }, 300);
    
    return () => clearTimeout(timer);
  }, [percentage]);
  
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.96 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
    >
      <Paper
        elevation={0}
        sx={{
          bgcolor: 'rgba(255, 255, 255, 0.9)',
          borderRadius: 3,
          boxShadow: `0 8px 32px ${alpha(color, 0.15)}`,
          border: `1px solid ${alpha(color, 0.2)}`,
          backdropFilter: 'blur(10px)',
          position: 'relative',
          overflow: 'hidden',
          transition: 'transform 0.2s ease, box-shadow 0.2s ease',
          '&:hover': {
            boxShadow: `0 12px 40px ${alpha(color, 0.2)}`,
            transform: 'translateY(-3px)'
          }
        }}
      >
        {/* Header */}
        <Box 
          sx={{ 
            p: 0.5, 
            bgcolor: alpha(color, 0.1),
            display: 'flex',
            justifyContent: 'center'
          }}
        >
          <Typography 
            sx={{ 
              fontSize: 12, 
              fontWeight: 600, 
              color,
              textTransform: 'uppercase',
              letterSpacing: 0.5
            }}
          >
            Success Probability
          </Typography>
        </Box>
        
        {/* Main content */}
        <Box 
          sx={{
            display: 'flex',
            flexDirection: { xs: 'column', sm: 'row' },
            alignItems: 'center',
            p: { xs: 3, sm: 4 },
            gap: { xs: 3, sm: 4 }
          }}
        >
          {/* Background gradient */}
          <Box 
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              backgroundImage: `
                radial-gradient(circle at top right, ${alpha(color, 0.06)} 0%, transparent 70%),
                linear-gradient(to bottom, ${alpha(color, 0.02)} 0%, transparent 100%)
              `,
              zIndex: 0
            }}
          />
          
          {/* Progress circle */}
          <Box
            sx={{
              position: 'relative',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 1,
              flexShrink: 0
            }}
          >
            <CircularProgress
              variant="determinate"
              value={100}
              size={isMobile ? 80 : 94}
              thickness={5}
              sx={{
                color: alpha(color, 0.12),
                position: 'absolute'
              }}
            />
            <CircularProgress
              variant="determinate"
              value={animatedProb}
              size={isMobile ? 80 : 94}
              thickness={5}
              sx={{
                color: color,
                transition: 'all 1.5s cubic-bezier(0.65, 0, 0.35, 1)'
              }}
            />
            <Box
              sx={{
                position: 'absolute',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexDirection: 'column'
              }}
            >
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.8, duration: 0.5 }}
              >
                <Typography
                  variant="h3"
                  sx={{
                    fontWeight: 700,
                    color,
                    fontSize: isMobile ? 26 : 32,
                    lineHeight: 1,
                    textAlign: 'center'
                  }}
                >
                  {Math.round(animatedProb)}
                  <Typography 
                    component="span" 
                    sx={{ 
                      fontWeight: 500, 
                      fontSize: isMobile ? 16 : 20,
                      verticalAlign: 'top',
                      position: 'relative',
                      top: -2,
                      left: 2,
                      opacity: 0.9
                    }}
                  >
                    %
                  </Typography>
                </Typography>
              </motion.div>
              
              <Box 
                sx={{ 
                  mt: 0.5, 
                  display: 'flex', 
                  alignItems: 'center', 
                  color,
                  fontSize: isMobile ? 12 : 14
                }}
              >
                {React.cloneElement(icon as React.ReactElement, { 
                  sx: { fontSize: isMobile ? 14 : 16, mr: 0.5 } 
                })}
                {category}
              </Box>
            </Box>
          </Box>
          
          {/* Text content */}
          <Box sx={{ zIndex: 1, flexGrow: 1, width: '100%' }}>            
            <Box sx={{ mb: 3 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Typography
                  sx={{
                    fontSize: 14,
                    fontWeight: 600,
                    color: '#64748B'
                  }}
                >
                  Investment Confidence
                </Typography>
                <Box 
                  sx={{ 
                    ml: 1.5,
                    px: 1.5, 
                    py: 0.3, 
                    bgcolor: alpha(color, 0.1),
                    borderRadius: 5,
                    display: 'flex',
                    alignItems: 'center'
                  }}
                >
                  <Box 
                    sx={{ 
                      width: 6, 
                      height: 6, 
                      borderRadius: '50%', 
                      bgcolor: color,
                      mr: 0.8
                    }} 
                  />
                  <Typography
                    sx={{
                      fontSize: 12,
                      fontWeight: 600,
                      color
                    }}
                  >
                    {category}
                  </Typography>
                </Box>
              </Box>
              
              <LinearProgress 
                variant="determinate" 
                value={animatedProb} 
                sx={{
                  height: 6,
                  borderRadius: 3,
                  bgcolor: alpha(color, 0.12),
                  '& .MuiLinearProgress-bar': {
                    bgcolor: color,
                    borderRadius: 3
                  }
                }}
              />
            </Box>
            
            <Box>
              {renderBulletPoint('Based on CAMP analysis scoring')}
              {renderBulletPoint('Benchmarked against 10,000+ startups')}
              {renderBulletPoint(`${getSuggestionByScore(percentage)}`)}
            </Box>
          </Box>
        </Box>
      </Paper>
    </motion.div>
  );
};

// Helper function to render bullet points
const renderBulletPoint = (text: string) => (
  <Box sx={{ display: 'flex', alignItems: 'flex-start', mb: 1 }}>
    <Box 
      sx={{ 
        width: 5, 
        height: 5, 
        borderRadius: '50%', 
        bgcolor: '#94A3B8', 
        mt: 1,
        mr: 1.5,
        flexShrink: 0
      }} 
    />
    <Typography
      variant="body2"
      sx={{
        color: '#64748B',
        fontSize: 13,
        lineHeight: 1.4
      }}
    >
      {text}
    </Typography>
  </Box>
);

// Get suggestion based on score
const getSuggestionByScore = (score: number): string => {
  if (score >= 80) return 'Top-tier investment opportunity';
  if (score >= 65) return 'Strong growth trajectory signals';
  if (score >= 50) return 'Good potential with execution focus';
  if (score >= 35) return 'Consider refinements to business model';
  return 'Significant restructuring recommended';
};

export default SummaryBadge;
