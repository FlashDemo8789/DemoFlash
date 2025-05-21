import React, { useState, useEffect } from "react";
import { 
  Box, 
  Typography, 
  Container, 
  Grid, 
  Paper, 
  Button,
  alpha,
  Divider,
  useMediaQuery,
  useTheme,
  Chip,
  Tab,
  Tabs,
  CircularProgress,
  Alert,
  Dialog,
  DialogContent
} from "@mui/material";
import { 
  ArrowBack as ArrowBackIcon,
  Download as DownloadIcon,
  Home as HomeIcon,
  Share as ShareIcon,
  Analytics as AnalyticsIcon
} from "@mui/icons-material";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import SummaryBadge from "../components/SummaryBadge";
import RadarView from "../components/RadarView";
import AlertsBox from "../components/AlertsBox";
import PDFDownloadButton from "../components/PDFDownloadButton";
import HierarchicalPDFButton from "../components/HierarchicalPDFButton";
import HierarchicalModelResults from "../components/HierarchicalModelResults";
import RecommendationsPanel from "../components/RecommendationsPanel";
import PredictionVisualization from "../components/PredictionVisualization";
import ModelInfoPanel from "../components/ModelInfoPanel";
import HierarchicalModelOnboarding from "../components/HierarchicalModelOnboarding";
import { 
  AnalysisResult, 
  PillarScores, 
  HierarchicalPredictionResponse,
  RecommendationsResponse 
} from "../types/api";
import {
  getPrediction,
  getRecommendations,
  getModelInfo,
  getHierarchicalAnalysis
} from "../services/hierarchicalModelService";
import { enableHierarchicalModel, showFeatureOnboarding } from "../constants/featureFlags";
import { 
  shouldShowHierarchicalOnboarding, 
  markHierarchicalOnboardingAsShown, 
  markHierarchicalOnboardingAsCompleted 
} from "../services/onboardingService";

interface ResultsPageProps {
  result: AnalysisResult;
}

const COLOR_PALETTE = {
  primary: "#3B82F6",
  surface: "#FFFFFF",
  surfaceHover: "#F9FAFC",
  textPrimary: "#111827",
  textSecondary: "#64748B",
  border: "rgba(229, 234, 243, 0.7)"
};

const ResultsPage: React.FC<ResultsPageProps> = ({ result }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [activeTab, setActiveTab] = useState<number>(0);
  const [showHierarchicalView, setShowHierarchicalView] = useState<boolean>(false);
  const [showVisualization, setShowVisualization] = useState<boolean>(false);
  
  // State for API data
  const [hierarchicalResult, setHierarchicalResult] = useState<HierarchicalPredictionResponse | null>(null);
  const [recommendations, setRecommendations] = useState<RecommendationsResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  // State for onboarding
  const [showOnboarding, setShowOnboarding] = useState<boolean>(false);
  
  // Extract important data with fallbacks
  const { 
    capital_score = 0, 
    advantage_score = 0, 
    market_score = 0, 
    people_score = 0,
    overall_score = 0,
    success_probability = 0, 
    pillar_scores = {} as PillarScores,
    alerts = [] 
  } = result;
  
  // Use pillar_scores if available, fallback to the individual scores
  const effectivePillarScores: PillarScores = {
    capital: pillar_scores?.capital ?? capital_score,
    advantage: pillar_scores?.advantage ?? advantage_score,
    market: pillar_scores?.market ?? market_score,
    people: pillar_scores?.people ?? people_score,
  };
  
  const startupId = result.startup_id || "demo";
  const startupName = result.startup_name || "Your Startup";
  
  // Load hierarchical model data when the active tab changes to the hierarchical view
  useEffect(() => {
    if (activeTab === 1 && !hierarchicalResult && !recommendations) {
      loadHierarchicalData();
    }
    
    // Check if onboarding should be shown when switching to hierarchical tab
    if (activeTab === 1 && enableHierarchicalModel && showFeatureOnboarding) {
      // Check user preference from localStorage
      const shouldShow = shouldShowHierarchicalOnboarding();
      
      if (shouldShow) {
        // Mark as shown for this session
        markHierarchicalOnboardingAsShown();
        setShowOnboarding(true);
      }
    }
  }, [activeTab]);
  
  // Function to load hierarchical model data
  const loadHierarchicalData = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Prepare data to send to backend
      // We need to convert the AnalysisResult to a simple metrics object
      // that the backend can process
      const metricsData = {
        startup_id: result.startup_id || "demo",
        startup_name: result.startup_name || "Demo Startup",
        sector: result.sector || "Technology",
        // Include all numeric metrics from the result
        ...Object.entries(result)
          .filter(([key, value]) => 
            typeof value === 'number' && 
            !['capital_score', 'advantage_score', 'market_score', 'people_score', 'overall_score', 'success_probability'].includes(key)
          )
          .reduce((obj, [key, value]) => ({ ...obj, [key]: value }), {})
      };

      console.log('Sending metrics data to API:', metricsData);
      
      const { prediction, recommendations } = await getHierarchicalAnalysis(metricsData);
      setHierarchicalResult(prediction);
      setRecommendations(recommendations);
      
      // Log success for debugging
      console.log('Successfully loaded hierarchical data');
      console.log('Prediction:', prediction);
      console.log('Recommendations:', recommendations);
    } catch (err) {
      console.error("Error loading hierarchical data:", err);
      setError(err instanceof Error ? err.message : "An unknown error occurred");
      
      // Create fallback data based on the traditional analysis
      if (!hierarchicalResult) {
        console.log('Using fallback data based on traditional analysis');
        setHierarchicalResult({
          pillar_scores: effectivePillarScores,
          final_score: success_probability,
          prediction: success_probability >= 0.5 ? 'pass' : 'fail',
          confidence: Math.max(success_probability, 1 - success_probability),
          threshold: 0.5
        });
      }
    } finally {
      setIsLoading(false);
    }
  };
  
  // Handle tab change
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };
  
  // Toggle visualization view
  const toggleVisualization = () => {
    setShowVisualization(!showVisualization);
  };
  
  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.15,
        delayChildren: 0.1,
      }
    }
  };
  
  const itemVariants = {
    hidden: { y: 15, opacity: 0 },
    visible: { 
      y: 0, 
      opacity: 1,
      transition: { duration: 0.5, ease: [0.22, 1, 0.36, 1] }
    }
  };
  
  // Render loading state
  const renderLoading = () => (
    <Box sx={{ 
      display: 'flex', 
      flexDirection: 'column',
      alignItems: 'center', 
      justifyContent: 'center',
      height: '200px',
      gap: 2
    }}>
      <CircularProgress size={40} />
      <Typography variant="body1" color="text.secondary">
        Loading hierarchical model data...
      </Typography>
    </Box>
  );
  
  // Render error state
  const renderError = () => (
    <Alert 
      severity="error" 
      sx={{ mb: 4 }}
      action={
        <Button 
          color="inherit" 
          size="small"
          onClick={loadHierarchicalData}
        >
          Retry
        </Button>
      }
    >
      {error}
    </Alert>
  );

  // Handle onboarding events
  const handleCloseOnboarding = () => {
    setShowOnboarding(false);
  };
  
  const handleCompleteOnboarding = () => {
    markHierarchicalOnboardingAsCompleted();
    setShowOnboarding(false);
  };

  return (
    <Box 
      sx={{ 
        minHeight: '100vh',
        backgroundImage: 'linear-gradient(135deg, #f5f7ff 0%, #ffffff 100%)',
        py: { xs: 4, md: 5 }
      }}
    >
      <Container maxWidth="xl">
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          <Box sx={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            mb: 4 
          }}>
            <Link to="/" style={{ textDecoration: 'none' }}>
              <Button 
                startIcon={<ArrowBackIcon />}
                variant="outlined"
                sx={{
                  borderRadius: 2,
                  textTransform: 'none',
                  px: 2,
                  py: 1,
                  color: COLOR_PALETTE.primary,
                  borderColor: alpha(COLOR_PALETTE.primary, 0.3),
                  fontSize: 14,
                  fontWeight: 500,
                  '&:hover': {
                    borderColor: COLOR_PALETTE.primary,
                    bgcolor: alpha(COLOR_PALETTE.primary, 0.04),
                  }
                }}
              >
                Back to Wizard
              </Button>
            </Link>
            
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button 
                startIcon={<ShareIcon />}
                sx={{
                  borderRadius: 2,
                  textTransform: 'none',
                  px: 2,
                  py: 1,
                  color: COLOR_PALETTE.textSecondary,
                  fontSize: 14,
                  fontWeight: 500,
                  '&:hover': {
                    bgcolor: alpha(COLOR_PALETTE.textSecondary, 0.04),
                  }
                }}
              >
                Share
              </Button>
              
              <Link to="/" style={{ textDecoration: 'none' }}>
                <Button 
                  startIcon={<HomeIcon />}
                  sx={{
                    borderRadius: 2,
                    textTransform: 'none',
                    px: 2,
                    py: 1,
                    color: COLOR_PALETTE.textSecondary,
                    fontSize: 14,
                    fontWeight: 500,
                    '&:hover': {
                      bgcolor: alpha(COLOR_PALETTE.textSecondary, 0.04),
                    }
                  }}
                >
                  Home
                </Button>
              </Link>
            </Box>
          </Box>
        </motion.div>
        
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          {/* Header */}
          <motion.div variants={itemVariants}>
            <Paper
              elevation={0}
              sx={{
                p: { xs: 3, md: 5 },
                mb: 4,
                borderRadius: 3,
                bgcolor: COLOR_PALETTE.surface,
                backdropFilter: 'blur(10px)',
                border: `1px solid ${COLOR_PALETTE.border}`,
                boxShadow: '0 5px 30px rgba(0, 0, 0, 0.03)',
                position: 'relative',
                overflow: 'hidden'
              }}
            >
              <Box 
                sx={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  background: 'radial-gradient(circle at top right, rgba(59, 130, 246, 0.03) 0%, transparent 70%)',
                  zIndex: 0
                }}
              />
              
              <Box sx={{ position: 'relative', zIndex: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Chip 
                    label="FlashDNA Analysis" 
                    size="small"
                    sx={{ 
                      bgcolor: alpha(COLOR_PALETTE.primary, 0.1),
                      color: COLOR_PALETTE.primary,
                      fontWeight: 600,
                      fontSize: 12,
                      height: 26,
                      mr: 2
                    }}
                  />
                  <Typography 
                    variant="body2" 
                    sx={{ 
                      color: COLOR_PALETTE.textSecondary, 
                      fontWeight: 500,
                    }}
                  >
                    Generated on {new Date().toLocaleDateString()}
                  </Typography>
                </Box>
                
                <Typography 
                  variant="h3" 
                  sx={{ 
                    fontWeight: 800, 
                    color: COLOR_PALETTE.textPrimary, 
                    letterSpacing: '-0.02em', 
                    mb: 1.5,
                    fontSize: { xs: 28, md: 36, lg: 42 },
                    lineHeight: 1.2
                  }}
                >
                  {startupName}{' '}
                  <Box 
                    component="span" 
                    sx={{ 
                      background: 'linear-gradient(90deg, #3B82F6, #8B5CF6)',
                      backgroundClip: 'text',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent'
                    }}
                  >
                    Investment Analysis
                  </Box>
                </Typography>
                
                <Typography 
                  variant="body1" 
                  sx={{ 
                    color: COLOR_PALETTE.textSecondary, 
                    maxWidth: 700,
                    lineHeight: 1.6,
                    fontSize: { xs: 15, md: 16 }
                  }}
                >
                  Here's your startup's proprietary Flash DNA investment analysis. Review the results, 
                  share these insights with your team or potential investors, and use the insights to
                  refine your business strategy.
                </Typography>
                
                {/* Toggle between traditional and hierarchical view */}
                <Box sx={{ mt: 3 }}>
                  <Tabs 
                    value={activeTab} 
                    onChange={handleTabChange}
                    indicatorColor="primary"
                    textColor="primary"
                    variant="fullWidth"
                    sx={{
                      '& .MuiTab-root': {
                        textTransform: 'none',
                        fontWeight: 600,
                        fontSize: 14,
                      }
                    }}
                  >
                    <Tab label="Traditional View" />
                    {enableHierarchicalModel && (
                      <Tab label="Hierarchical Model" />
                    )}
                  </Tabs>
                </Box>
              </Box>
            </Paper>
          </motion.div>
          
          {/* Error display */}
          {error && activeTab === 1 && enableHierarchicalModel && renderError()}
          
          {/* Main content */}
          {activeTab === 0 || !enableHierarchicalModel ? (
            /* Traditional View */
            <Grid container spacing={4}>
              {/* Left column */}
              <Grid item xs={12} md={4}>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  <motion.div variants={itemVariants}>
                    <SummaryBadge prob={success_probability} />
                  </motion.div>
                  
                  <motion.div variants={itemVariants}>
                    <Paper
                      elevation={0}
                      sx={{
                        borderRadius: 3,
                        bgcolor: COLOR_PALETTE.surface,
                        backdropFilter: 'blur(10px)',
                        border: `1px solid ${COLOR_PALETTE.border}`,
                        boxShadow: '0 5px 30px rgba(0, 0, 0, 0.03)',
                        overflow: 'hidden',
                        transition: 'transform 0.2s ease, box-shadow 0.2s ease',
                        '&:hover': {
                          boxShadow: '0 8px 35px rgba(0, 0, 0, 0.05)',
                          transform: 'translateY(-2px)'
                        }
                      }}
                    >
                      <Box sx={{ p: 0.5, bgcolor: alpha('#F59E0B', 0.1) }}>
                        <Typography 
                          align="center" 
                          sx={{ 
                            fontSize: 12, 
                            fontWeight: 600, 
                            color: '#F59E0B',
                            textTransform: 'uppercase',
                            letterSpacing: 0.5
                          }}
                        >
                          Action Items
                        </Typography>
                      </Box>
                      <Box sx={{ p: { xs: 3, md: 4 } }}>
                        <AlertsBox alerts={alerts} />
                      </Box>
                    </Paper>
                  </motion.div>
                  
                  <motion.div variants={itemVariants}>
                    <PDFDownloadButton startupId={startupId} payload={result} />
                  </motion.div>
                </Box>
              </Grid>
              
              {/* Right column */}
              <Grid item xs={12} md={8}>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  <motion.div variants={itemVariants}>
                    <RadarView 
                      capital_score={capital_score}
                      advantage_score={advantage_score}
                      market_score={market_score}
                      people_score={people_score}
                      pillar_scores={pillar_scores}
                      overall_score={overall_score}
                    />
                  </motion.div>
                  
                  <motion.div variants={itemVariants}>
                    <Paper
                      elevation={0}
                      sx={{
                        borderRadius: 3,
                        bgcolor: COLOR_PALETTE.surface,
                        backdropFilter: 'blur(10px)',
                        border: `1px solid ${COLOR_PALETTE.border}`,
                        boxShadow: '0 5px 30px rgba(0, 0, 0, 0.03)',
                        overflow: 'hidden',
                        transition: 'transform 0.2s ease, box-shadow 0.2s ease',
                        '&:hover': {
                          boxShadow: '0 8px 35px rgba(0, 0, 0, 0.05)',
                          transform: 'translateY(-2px)'
                        }
                      }}
                    >
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
                          Key Metrics
                        </Typography>
                      </Box>
                      <Box sx={{ p: { xs: 3, md: 4 } }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                          <AnalyticsIcon sx={{ color: COLOR_PALETTE.primary, mr: 1.5 }} />
                          <Typography 
                            variant="h6" 
                            sx={{ 
                              fontWeight: 600,
                              color: COLOR_PALETTE.textPrimary
                            }}
                          >
                            Investment Analysis Overview
                          </Typography>
                        </Box>
                        
                        <Grid container spacing={3}>
                          {Object.entries(result)
                            .filter(([key]) => {
                              // Filter out keys we're already displaying elsewhere and internal data
                              const excludeKeys = ['pillar_scores', 'success_probability', 'alerts', 'startup_id', 'flashdna_score'];
                              return !excludeKeys.includes(key) && key !== 'alerts' && !key.startsWith('_');
                            })
                            .slice(0, 6) // Limit to prevent overwhelming the UI
                            .map(([key, value]) => (
                              <Grid item xs={12} sm={6} key={key}>
                                <Paper
                                  elevation={0}
                                  sx={{
                                    p: 2.5,
                                    borderRadius: 2,
                                    bgcolor: COLOR_PALETTE.surfaceHover,
                                    border: `1px solid ${COLOR_PALETTE.border}`,
                                    transition: 'all 0.2s ease',
                                    '&:hover': {
                                      boxShadow: '0 4px 12px rgba(0, 0, 0, 0.04)',
                                    }
                                  }}
                                >
                                  <Typography 
                                    variant="body2" 
                                    sx={{ 
                                      color: COLOR_PALETTE.textSecondary, 
                                      fontWeight: 500,
                                      mb: 0.5,
                                      textTransform: 'capitalize',
                                      fontSize: 13
                                    }}
                                  >
                                    {key.replace(/_/g, ' ')}
                                  </Typography>
                                  <Typography 
                                    variant="body1" 
                                    sx={{ 
                                      fontWeight: 600,
                                      color: COLOR_PALETTE.textPrimary,
                                      fontSize: 18
                                    }}
                                  >
                                    {typeof value === 'number' ? 
                                      Number.isInteger(value) ? value : value.toFixed(2) : 
                                      String(value)}
                                  </Typography>
                                </Paper>
                              </Grid>
                            ))}
                        </Grid>
                        
                        <Divider sx={{ my: 3 }} />
                        
                        <Box sx={{ textAlign: 'center' }}>
                          <Button
                            variant="outlined"
                            sx={{
                              borderRadius: 2,
                              textTransform: 'none',
                              px: 3,
                              py: 1,
                              color: COLOR_PALETTE.primary,
                              borderColor: alpha(COLOR_PALETTE.primary, 0.3),
                              fontSize: 14,
                              fontWeight: 500,
                              '&:hover': {
                                borderColor: COLOR_PALETTE.primary,
                                bgcolor: alpha(COLOR_PALETTE.primary, 0.04),
                              }
                            }}
                          >
                            View All Metrics
                          </Button>
                        </Box>
                      </Box>
                    </Paper>
                  </motion.div>
                </Box>
              </Grid>
            </Grid>
          ) : isLoading ? (
            /* Loading state for hierarchical view */
            renderLoading()
          ) : (
            /* Hierarchical Model View */
            <Grid container spacing={4}>
              {/* Left column */}
              <Grid item xs={12} md={6}>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  <motion.div variants={itemVariants}>
                    {hierarchicalResult && (
                      <HierarchicalModelResults 
                        prediction={hierarchicalResult}
                        showVisualization={true}
                        onViewVisualization={toggleVisualization}
                      />
                    )}
                  </motion.div>
                  
                  <motion.div variants={itemVariants}>
                    <RecommendationsPanel 
                      recommendations={recommendations} 
                      isLoading={isLoading && !recommendations}
                      error={error && !recommendations ? error : null} 
                    />
                  </motion.div>
                </Box>
              </Grid>
              
              {/* Right column */}
              <Grid item xs={12} md={6}>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  {showVisualization ? (
                    <motion.div variants={itemVariants}>
                      <PredictionVisualization 
                        startupData={result} 
                        isLoading={isLoading}
                        error={error} 
                      />
                    </motion.div>
                  ) : (
                    <motion.div variants={itemVariants}>
                      <ModelInfoPanel 
                        isLoading={isLoading} 
                        error={error}
                      />
                    </motion.div>
                  )}
                  
                  <motion.div variants={itemVariants}>
                    <HierarchicalPDFButton 
                      startupId={startupId} 
                      payload={result}
                      hierarchicalData={{
                        prediction: hierarchicalResult,
                        recommendations: recommendations
                      }}
                    />
                  </motion.div>
                </Box>
              </Grid>
            </Grid>
          )}
        </motion.div>
      </Container>
      
      {/* Onboarding Dialog */}
      <Dialog
        open={showOnboarding}
        onClose={handleCloseOnboarding}
        maxWidth="md"
        fullWidth
        PaperProps={{
          elevation: 0,
          sx: { borderRadius: 3, overflow: 'visible' }
        }}
      >
        <DialogContent sx={{ p: 0 }}>
          <HierarchicalModelOnboarding
            onClose={handleCloseOnboarding}
            onComplete={handleCompleteOnboarding}
          />
        </DialogContent>
      </Dialog>
    </Box>
  );
};

export default ResultsPage;
