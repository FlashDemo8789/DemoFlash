import React, { useState } from "react";
import { 
  Button, 
  Box, 
  CircularProgress, 
  Typography, 
  Paper,
  alpha, 
  Collapse,
  useTheme,
  useMediaQuery,
  Snackbar,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControlLabel,
  Checkbox,
  Divider
} from "@mui/material";
import { 
  Download as DownloadIcon, 
  PictureAsPdf as PdfIcon,
  Error as ErrorIcon,
  Check as CheckIcon,
  Article as ArticleIcon
} from "@mui/icons-material";
import { motion } from "framer-motion";
import { AnalysisResult, HierarchicalPredictionResponse, RecommendationsResponse } from '../types/api';
import { getVisualizationBlob } from '../services/hierarchicalModelService';
import { enableHierarchicalPdfExport } from '../constants/featureFlags';

// Define local types to replace the imported ones
// This avoids issues with TypeScript imports
interface MetricValues {
  startup_name?: string;
  startup_id?: string;
  [key: string]: any;
}

type FormData = MetricValues;

interface PDFDownloadButtonProps {
  startupId: string;
  payload: AnalysisResult;
  hierarchicalData?: {
    prediction: HierarchicalPredictionResponse | null;
    recommendations: RecommendationsResponse | null;
  };
  isHierarchicalView?: boolean;
}

const COLOR_PALETTE = {
  primary: "#3B82F6",
  success: "#10B981",
  error: "#EF4444",
  surface: "#FFFFFF",
  surfaceHover: "#F9FAFC",
  textPrimary: "#111827",
  textSecondary: "#64748B",
  border: "rgba(229, 234, 243, 0.7)"
};

const PDFDownloadButton: React.FC<PDFDownloadButtonProps> = ({
  startupId,
  payload,
  hierarchicalData,
  isHierarchicalView = false
}) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [showOptions, setShowOptions] = useState(false);
  const [includeHierarchical, setIncludeHierarchical] = useState(isHierarchicalView);
  const [includeVisualization, setIncludeVisualization] = useState(true);
  const [includeRecommendations, setIncludeRecommendations] = useState(true);
  
  // Check if hierarchical data is available
  const hasHierarchicalData = !!(
    hierarchicalData?.prediction && 
    hierarchicalData?.recommendations &&
    enableHierarchicalPdfExport
  );

  // Function to generate PDF with hierarchical data
  const generateHierarchicalPDF = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // If we need to include visualization, get the visualization image
      let visualizationBlob = null;
      if (includeVisualization && hierarchicalData?.prediction) {
        try {
          // Prepare data for visualization
          const metricsData = {
            startup_id: payload.startup_id || "demo",
            startup_name: payload.startup_name || "Demo Startup",
            sector: payload.sector || "Technology",
            // Include all numeric metrics from the result
            ...Object.entries(payload)
              .filter(([key, value]) => 
                typeof value === 'number' && 
                !['capital_score', 'advantage_score', 'market_score', 'people_score', 'overall_score', 'success_probability'].includes(key)
              )
              .reduce((obj, [key, value]) => ({ ...obj, [key]: value }), {})
          };
          
          visualizationBlob = await getVisualizationBlob(metricsData);
        } catch (err) {
          console.error('Error getting visualization for PDF:', err);
          // Continue without visualization if it fails
        }
      }
      
      // Create FormData object to send to API
      const formData = new FormData();
      formData.append('startup_id', startupId);
      formData.append('analysis_data', JSON.stringify(payload));
      
      // Add hierarchical data if available and requested
      if (includeHierarchical && hierarchicalData?.prediction) {
        formData.append('hierarchical_data', JSON.stringify({
          prediction: hierarchicalData.prediction,
          recommendations: includeRecommendations ? hierarchicalData.recommendations : null
        }));
      }
      
      // Add visualization if available
      if (visualizationBlob) {
        formData.append('visualization', visualizationBlob, 'visualization.png');
      }

      // Send to API
      const response = await fetch('/api/reports/generate-pdf', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Error generating PDF: ${response.statusText}`);
      }
      
      // Get the PDF blob and create a download link
      const pdfBlob = await response.blob();
      const url = URL.createObjectURL(pdfBlob);
      const link = document.createElement('a');
      
      // Generate filename with timestamp
      const timestamp = new Date().toISOString().split('T')[0];
      const fileName = `flash_analysis_${startupId}_${timestamp}.pdf`;
      
      // Download the PDF
      link.href = url;
      link.download = fileName;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      URL.revokeObjectURL(url);
      setSuccess(true);
      
    } catch (err) {
      console.error('Error downloading PDF:', err);
      setError(err instanceof Error ? err.message : 'Error generating PDF');
    } finally {
      setLoading(false);
    }
  };
  
  // Use traditional PDF generation as fallback
  const generateTraditionalPDF = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Create FormData object to send to API
      const formData = new FormData();
      formData.append('startup_id', startupId);
      formData.append('analysis_data', JSON.stringify(payload));
      
      // Send to API
      const response = await fetch('/api/reports/generate-pdf', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Error generating PDF: ${response.statusText}`);
      }
      
      // Get the PDF blob and create a download link
      const pdfBlob = await response.blob();
      const url = URL.createObjectURL(pdfBlob);
      const link = document.createElement('a');
      
      // Generate filename with timestamp
      const timestamp = new Date().toISOString().split('T')[0];
      const fileName = `flash_analysis_${startupId}_${timestamp}.pdf`;
      
      // Download the PDF
      link.href = url;
      link.download = fileName;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      URL.revokeObjectURL(url);
      setSuccess(true);
      
    } catch (err) {
      console.error('Error downloading PDF:', err);
      setError(err instanceof Error ? err.message : 'Error generating PDF');
    } finally {
      setLoading(false);
    }
  };
  
  // Handle download button click
  const handleDownload = () => {
    // If hierarchical data is available, show options dialog
    if (hasHierarchicalData) {
      setShowOptions(true);
    } else {
      // Otherwise, use traditional PDF generation
      generateTraditionalPDF();
    }
  };
  
  // Handle confirmation in options dialog
  const handleConfirmOptions = () => {
    setShowOptions(false);
    
    if (includeHierarchical) {
      generateHierarchicalPDF();
    } else {
      generateTraditionalPDF();
    }
  };
  
  // Close dialogs and alerts
  const handleCloseOptions = () => {
    setShowOptions(false);
  };
  
  const handleCloseSnackbar = () => {
    setSuccess(false);
    setError(null);
  };
  
  return (
    <>
      <Paper
        elevation={0}
        sx={{
          borderRadius: 3,
          bgcolor: 'rgba(255, 255, 255, 0.9)',
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
        {/* Header */}
        <Box sx={{ p: 0.5, bgcolor: alpha(COLOR_PALETTE.primary, 0.1) }}>
          <Typography 
            align="center" 
            sx={{ 
              fontSize: 12, 
              fontWeight: 600, 
              color: COLOR_PALETTE.primary,
              textTransform: 'uppercase',
              letterSpacing: 0.5
            }}
          >
            Download Analysis
          </Typography>
        </Box>
        
        <Box sx={{ p: { xs: 3, sm: 4 } }}>
          <Box sx={{ mb: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1.5 }}>
              <ArticleIcon sx={{ color: COLOR_PALETTE.primary, mr: 1.5 }} />
              <Typography 
                variant="h6" 
                sx={{ 
                  fontSize: 18,
                  fontWeight: 600,
                  color: COLOR_PALETTE.textPrimary
                }}
              >
                FlashDNA Report
              </Typography>
            </Box>
            
            <Typography
              variant="body2"
              sx={{
                color: COLOR_PALETTE.textSecondary,
                mb: 3,
                fontSize: 14,
                lineHeight: 1.6
              }}
            >
              Export a comprehensive PDF report with all metrics, CAMP analysis scores, success probability, 
              and actionable recommendations for your startup.
            </Typography>
          </Box>
          
          <motion.div
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <Button
              variant="contained"
              onClick={handleDownload}
              disabled={loading}
              fullWidth
              size="large"
              startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <PdfIcon />}
              endIcon={!loading && <DownloadIcon />}
              sx={{
                borderRadius: 2,
                py: 1.5,
                px: 2,
                fontWeight: 600,
                fontSize: 15,
                textTransform: 'none',
                background: `linear-gradient(45deg, ${COLOR_PALETTE.primary}, #6366F1)`,
                boxShadow: `0 4px 14px ${alpha(COLOR_PALETTE.primary, 0.3)}`,
                '&:hover': {
                  boxShadow: `0 6px 20px ${alpha(COLOR_PALETTE.primary, 0.4)}`,
                },
                '&:disabled': {
                  background: `linear-gradient(45deg, ${alpha(COLOR_PALETTE.primary, 0.7)}, ${alpha('#6366F1', 0.7)})`,
                }
              }}
            >
              {loading ? "Generating PDF..." : "Download Full Report"}
            </Button>
          </motion.div>
          
          <Box sx={{ mt: 3, minHeight: 40 }}>
            <Collapse in={!!error} unmountOnExit>
              <motion.div
                initial={{ opacity: 0, y: -5 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -5 }}
                transition={{ duration: 0.3 }}
              >
                <Box
                  sx={{
                    p: 2,
                    borderRadius: 2,
                    bgcolor: alpha(COLOR_PALETTE.error, 0.08),
                    border: `1px solid ${alpha(COLOR_PALETTE.error, 0.2)}`,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1.5
                  }}
                >
                  <ErrorIcon 
                    sx={{ 
                      color: COLOR_PALETTE.error,
                      fontSize: 20 
                    }} 
                  />
                  <Typography 
                    sx={{ 
                      color: alpha(COLOR_PALETTE.error, 0.9), 
                      fontSize: 13, 
                      fontWeight: 500,
                      lineHeight: 1.4
                    }}
                  >
                    {error}
                  </Typography>
                </Box>
              </motion.div>
            </Collapse>
            
            <Collapse in={success} unmountOnExit>
              <motion.div
                initial={{ opacity: 0, y: -5 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -5 }}
                transition={{ duration: 0.3 }}
              >
                <Box
                  sx={{
                    p: 2,
                    borderRadius: 2,
                    bgcolor: alpha(COLOR_PALETTE.success, 0.08),
                    border: `1px solid ${alpha(COLOR_PALETTE.success, 0.2)}`,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1.5
                  }}
                >
                  <CheckIcon 
                    sx={{ 
                      color: COLOR_PALETTE.success,
                      fontSize: 20
                    }} 
                  />
                  <Typography 
                    sx={{ 
                      color: alpha(COLOR_PALETTE.success, 0.9), 
                      fontSize: 13, 
                      fontWeight: 500 
                    }}
                  >
                    PDF report downloaded successfully!
                  </Typography>
                </Box>
              </motion.div>
            </Collapse>
          </Box>
          
          {/* Report content list */}
          <Box sx={{ mt: 3, pt: 3, borderTop: `1px solid ${alpha(COLOR_PALETTE.border, 0.7)}` }}>
            <Typography 
              sx={{ 
                fontSize: 13, 
                fontWeight: 600, 
                color: COLOR_PALETTE.textSecondary,
                mb: 1.5
              }}
            >
              The report includes:
            </Typography>
            
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              {[
                'Executive Summary',
                'CAMP Analysis Breakdown', 
                'Success Probability Indicators',
                'Competitive Positioning',
                'Action Items & Recommendations'
              ].map((item, idx) => (
                <Box 
                  key={idx} 
                  sx={{ 
                    display: 'flex', 
                    alignItems: 'center',
                    gap: 1.5
                  }}
                >
                  <Box 
                    sx={{ 
                      width: 18, 
                      height: 18, 
                      borderRadius: '50%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      bgcolor: alpha(COLOR_PALETTE.primary, 0.1),
                      color: COLOR_PALETTE.primary,
                      fontSize: 12,
                      fontWeight: 600
                    }}
                  >
                    {idx + 1}
                  </Box>
                  <Typography 
                    sx={{ 
                      fontSize: 13, 
                      color: COLOR_PALETTE.textPrimary
                    }}
                  >
                    {item}
                  </Typography>
                </Box>
              ))}
            </Box>
          </Box>
        </Box>
      </Paper>
      
      {/* Options Dialog */}
      <Dialog open={showOptions} onClose={handleCloseOptions}>
        <DialogTitle>PDF Export Options</DialogTitle>
        <DialogContent>
          <Box sx={{ minWidth: 300, pt: 1 }}>
            <Typography variant="body2" color="text.secondary" paragraph>
              Choose which content to include in your PDF report:
            </Typography>
            
            <Divider sx={{ my: 2 }} />
            
            <FormControlLabel
              control={
                <Checkbox
                  checked={includeHierarchical}
                  onChange={(e) => setIncludeHierarchical(e.target.checked)}
                />
              }
              label="Include Hierarchical Model Results"
            />
            
            {includeHierarchical && (
              <Box sx={{ ml: 4, mt: 1 }}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={includeVisualization}
                      onChange={(e) => setIncludeVisualization(e.target.checked)}
                    />
                  }
                  label="Include Visualizations"
                />
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={includeRecommendations}
                      onChange={(e) => setIncludeRecommendations(e.target.checked)}
                    />
                  }
                  label="Include Recommendations"
                />
              </Box>
            )}
            
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              The PDF will always include traditional analysis results.
            </Typography>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseOptions}>Cancel</Button>
          <Button onClick={handleConfirmOptions} variant="contained">
            Generate PDF
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Success/Error Notifications */}
      <Snackbar
        open={success}
        autoHideDuration={5000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={handleCloseSnackbar} severity="success">
          PDF generated successfully!
        </Alert>
      </Snackbar>
      
      <Snackbar
        open={!!error}
        autoHideDuration={5000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={handleCloseSnackbar} severity="error">
          {error}
        </Alert>
      </Snackbar>
    </>
  );
};

export default PDFDownloadButton;
