import React from "react";
import {
  Box,
  Typography,
  Paper,
  alpha,
  useTheme,
  useMediaQuery,
  Chip,
  Divider
} from "@mui/material";
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
  Legend
} from "recharts";
import { 
  DonutLarge as ChartIcon,
  TrendingUp as TrendingUpIcon
} from "@mui/icons-material";
import { motion } from "framer-motion";
import { PillarScores } from '../types/api';

// Define local types for metrics scores with API field names
interface MetricScores {
  capital_score?: number;
  advantage_score?: number; 
  market_score?: number;
  people_score?: number;
  overall_score?: number;
  // Include legacy field names for backward compatibility
  capital?: number;
  advantage?: number; 
  market?: number;
  people?: number;
  flashdna_score?: number;
  [key: string]: number | undefined;
}

interface RadarViewProps {
  // Support both old and new API response formats
  capital_score?: number;
  advantage_score?: number;
  market_score?: number;
  people_score?: number;
  pillar_scores?: PillarScores;
  overall_score?: number;
  size?: 'small' | 'medium' | 'large';
}

// Pillar color palette
const PILLAR_COLORS = {
  capital: "#10B981",
  advantage: "#8B5CF6",
  market: "#F43F5E",
  people: "#F59E0B"
};

// Custom tooltip component for the radar chart
const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    
    return (
      <Paper
        elevation={3}
        sx={{
          bgcolor: 'rgba(255, 255, 255, 0.95)',
          p: 1.5,
          border: '1px solid #e5eaf3',
          borderRadius: 1.5,
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.08)'
        }}
      >
        <Typography 
          sx={{ 
            fontWeight: 600, 
            color: getPillarColor(data.pillar.toLowerCase()), 
            mb: 0.5,
            fontSize: 14
          }}
        >
          {data.pillar}
        </Typography>
        <Typography 
          sx={{ 
            fontSize: 18, 
            fontWeight: 700, 
            color: '#111827' 
          }}
        >
          {data.score} / 100
        </Typography>
      </Paper>
    );
  }
  return null;
};

// Helper to get pillar color
const getPillarColor = (pillar: string): string => {
  const key = pillar.toLowerCase().replace('_score', '');
  return PILLAR_COLORS[key as keyof typeof PILLAR_COLORS] || "#3B82F6";
};

// Function to convert API scores (0-1 range) to display scores (0-100 range)
const convertToDisplayScore = (score: number | undefined): number => {
  if (typeof score !== 'number') return 0;
  // Assume API scores are in 0-1 range, convert to 0-100
  return Math.round(score * 100);
};

// Function to normalize the scores object to expected format
const normalizeScores = (scores: MetricScores) => {
  return {
    capital: convertToDisplayScore(scores.capital_score || scores.capital),
    advantage: convertToDisplayScore(scores.advantage_score || scores.advantage),
    market: convertToDisplayScore(scores.market_score || scores.market),
    people: convertToDisplayScore(scores.people_score || scores.people),
    flashdna_score: convertToDisplayScore(scores.overall_score || scores.flashdna_score),
  };
};

const RadarView: React.FC<RadarViewProps> = ({
  capital_score,
  advantage_score,
  market_score,
  people_score,
  pillar_scores,
  overall_score,
  size = 'medium',
}) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  
  // Handle both old and new API formats
  const capitalScore = pillar_scores?.capital ?? capital_score ?? 0;
  const advantageScore = pillar_scores?.advantage ?? advantage_score ?? 0;
  const marketScore = pillar_scores?.market ?? market_score ?? 0;
  const peopleScore = pillar_scores?.people ?? people_score ?? 0;
  
  // Scale based on component size
  const getSize = () => {
    switch (size) {
      case 'small': return { height: 180, width: 180 };
      case 'large': return { height: 350, width: 350 };
      default: return { height: 250, width: 250 };
    }
  };
  
  const { height, width } = getSize();

  const scoreValues = {
    capital: capitalScore,
    advantage: advantageScore,
    market: marketScore,
    people: peopleScore
  };

  // Create radar data for the chart
  const radarData = [
    { pillar: 'Capital', score: Math.round(capitalScore * 100) },
    { pillar: 'Advantage', score: Math.round(advantageScore * 100) },
    { pillar: 'Market', score: Math.round(marketScore * 100) },
    { pillar: 'People', score: Math.round(peopleScore * 100) }
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
    >
      <Paper
        elevation={0}
        sx={{
          borderRadius: 3,
          bgcolor: 'rgba(255, 255, 255, 0.9)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(229,234,243,0.7)',
          boxShadow: '0 5px 30px rgba(0, 0, 0, 0.03)',
          overflow: 'hidden',
          transition: 'transform 0.2s ease, box-shadow 0.2s ease',
          '&:hover': {
            boxShadow: '0 8px 35px rgba(0, 0, 0, 0.05)',
            transform: 'translateY(-2px)'
          }
        }}
      >
        {/* Header */}
        <Box sx={{ p: 0.5, bgcolor: alpha('#3B82F6', 0.1) }}>
          <Typography 
            align="center" 
            sx={{ 
              fontSize: 12, 
              fontWeight: 600, 
              color: '#3B82F6',
              textTransform: 'uppercase',
              letterSpacing: 0.5
            }}
          >
            CAMP Analysis Radar
          </Typography>
        </Box>
        
        <Box sx={{ p: { xs: 2, md: 3 } }}>
          {/* Title with flash score */}
          <Box sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'space-between',
            mb: 2 
          }}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <ChartIcon sx={{ color: '#3B82F6', mr: 1 }} />
              <Typography 
                variant="h6" 
                sx={{ 
                  fontSize: { xs: 16, md: 18 },
                  fontWeight: 600,
                  color: '#111827'
                }}
              >
                Pillar Performance
              </Typography>
            </Box>
          </Box>
          
          {/* Radar chart */}
          <Box 
            sx={{ 
              height: { xs: 300, md: 350 }, 
              mt: 1,
              position: 'relative'
            }}
          >
            {/* Background elements */}
            <Box 
              sx={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                backgroundImage: `
                  radial-gradient(circle at center, ${alpha('#3B82F6', 0.02)} 0%, transparent 70%)
                `,
                zIndex: 0
              }}
            />
            
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData} outerRadius={isMobile ? 100 : 130}>
                <PolarGrid 
                  gridType="polygon" 
                  stroke="#d1d5db" 
                  strokeWidth={0.5}
                  strokeDasharray="3 3" 
                />
                <PolarAngleAxis 
                  dataKey="pillar" 
                  tick={{ 
                    fill: '#1f2937', 
                    fontSize: isMobile ? 12 : 14, 
                    fontWeight: 600
                  }} 
                  tickLine={false}
                  axisLine={{ stroke: '#E2E8F0' }}
                />
                <PolarRadiusAxis 
                  angle={45} 
                  domain={[0, 100]} 
                  axisLine={false}
                  tick={{ fill: '#64748B', fontSize: 11 }}
                  tickCount={5}
                  stroke="#E2E8F0"
                />
                <Radar 
                  name="Score" 
                  dataKey="score" 
                  stroke="#3B82F6" 
                  fill="#3B82F6" 
                  fillOpacity={0.6}
                  strokeWidth={2}
                  dot={{ fill: '#3B82F6', r: 4 }}
                  activeDot={{ r: 6, fill: '#3B82F6', stroke: 'white', strokeWidth: 2 }}
                  animationBegin={300}
                  animationDuration={1200}
                  animationEasing="ease-out"
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend 
                  wrapperStyle={{ 
                    fontSize: 12, 
                    fontWeight: 500,
                    color: '#64748B',
                    paddingTop: 10
                  }} 
                />
              </RadarChart>
            </ResponsiveContainer>
          </Box>
          
          <Divider sx={{ my: 2, opacity: 0.6 }} />
          
          {/* Pillar score cards */}
          <Typography 
            variant="subtitle2" 
            sx={{ 
              mb: 2, 
              color: '#64748B',
              fontSize: 13,
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              '&::before': {
                content: '""',
                display: 'block',
                width: 14,
                height: 2,
                bgcolor: '#94A3B8',
                mr: 1,
                borderRadius: 1
              }
            }}
          >
            PILLAR BREAKDOWN
          </Typography>
          
          <Box sx={{ 
            display: 'flex', 
            flexWrap: 'wrap',
            gap: 2
          }}>
            {Object.entries(scoreValues)
              .map(([key, value]) => (
                <Paper
                  key={key}
                  elevation={0}
                  sx={{
                    p: 2,
                    borderRadius: 2,
                    bgcolor: alpha(getPillarColor(key), 0.05),
                    border: `1px solid ${alpha(getPillarColor(key), 0.15)}`,
                    minWidth: { xs: 'calc(50% - 8px)', md: 120 },
                    flexGrow: { xs: 1, md: 0 },
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    transition: 'all 0.2s ease',
                    '&:hover': {
                      transform: 'translateY(-2px)',
                      boxShadow: `0 4px 12px ${alpha(getPillarColor(key), 0.08)}`,
                      bgcolor: alpha(getPillarColor(key), 0.08),
                    }
                  }}
                >
                  <Typography sx={{ 
                    fontSize: 12, 
                    fontWeight: 600, 
                    color: getPillarColor(key),
                    textTransform: 'uppercase',
                    mb: 0.5
                  }}>
                    {key[0].toUpperCase() + key.slice(1)}
                  </Typography>
                  <Typography sx={{ 
                    fontSize: { xs: 20, md: 24 }, 
                    fontWeight: 700, 
                    color: getPillarColor(key),
                    lineHeight: 1
                  }}>
                    {Math.round(value * 100)}
                  </Typography>
                  <Typography sx={{
                    fontSize: 11,
                    color: '#64748B',
                    mt: 0.5
                  }}>
                    out of 100
                  </Typography>
                </Paper>
              ))
            }
          </Box>
        </Box>
      </Paper>
    </motion.div>
  );
};

export default RadarView;
