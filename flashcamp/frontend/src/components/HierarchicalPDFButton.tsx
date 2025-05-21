import React, { useState } from 'react';
import {
  Button,
  CircularProgress,
  Snackbar,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControlLabel,
  Checkbox,
  Box,
  Typography,
  Divider
} from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';
import { 
  AnalysisResult, 
  HierarchicalPredictionResponse, 
  RecommendationsResponse 
} from '../types/api';
import { getVisualizationBlob } from '../services/hierarchicalModelService';
import { enableHierarchicalPdfExport } from '../constants/featureFlags';

interface HierarchicalPDFButtonProps {
  startupId: string;
  payload: AnalysisResult;
  hierarchicalData: {
    prediction: HierarchicalPredictionResponse | null;
    recommendations: RecommendationsResponse | null;
  };
}

const HierarchicalPDFButton: React.FC<HierarchicalPDFButtonProps> = ({
  startupId,
  payload,
  hierarchicalData
}) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [showOptions, setShowOptions] = useState(false);
  const [includeVisualization, setIncludeVisualization] = useState(true);
  const [includeRecommendations, setIncludeRecommendations] = useState(true);
  
  // Check if hierarchical data is available and feature flag is enabled
  const isEnabled = !!(
    hierarchicalData?.prediction && 
    hierarchicalData?.recommendations &&
    enableHierarchicalPdfExport
  );

  // Generate PDF with hierarchical model data
  const generatePDF = async () => {
    if (!isEnabled) {
      setError('Hierarchical model PDF export is not available');
      return;
    }
    
    try {
      setLoading(true);
      setError(null);
      
      // If visualization is requested, get the visualization blob
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
      
      // Create FormData object to send to the server
      const formData = new FormData();
      formData.append('startup_id', startupId);
      formData.append('analysis_data', JSON.stringify(payload));
      
      // Add hierarchical data
      formData.append('hierarchical_data', JSON.stringify({
        prediction: hierarchicalData.prediction,
        recommendations: includeRecommendations ? hierarchicalData.recommendations : null
      }));
      
      // Add visualization if available
      if (visualizationBlob) {
        formData.append('visualization', visualizationBlob, 'visualization.png');
      }
      
      // Send to the server
      const response = await fetch('/api/reports/generate-pdf', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Error generating PDF: ${response.statusText}`);
      }
      
      // Get the PDF blob and download it
      const pdfBlob = await response.blob();
      const url = URL.createObjectURL(pdfBlob);
      const link = document.createElement('a');
      
      // Generate filename with timestamp
      const timestamp = new Date().toISOString().split('T')[0];
      const fileName = `flash_hierarchical_analysis_${startupId}_${timestamp}.pdf`;
      
      // Download the PDF
      link.href = url;
      link.download = fileName;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      URL.revokeObjectURL(url);
      setSuccess(true);
      
    } catch (err) {
      console.error('Error generating hierarchical PDF:', err);
      setError(err instanceof Error ? err.message : 'Error generating PDF');
    } finally {
      setLoading(false);
    }
  };
  
  // Handle button click
  const handleDownload = () => {
    if (isEnabled) {
      setShowOptions(true);
    } else {
      setError('Hierarchical model PDF export is not available');
    }
  };
  
  // Dialog handlers
  const handleConfirmOptions = () => {
    setShowOptions(false);
    generatePDF();
  };
  
  const handleCloseOptions = () => {
    setShowOptions(false);
  };
  
  const handleCloseSnackbar = () => {
    setSuccess(false);
    setError(null);
  };
  
  // If feature is disabled, don't render the button
  if (!isEnabled) {
    return null;
  }
  
  return (
    <>
      <Button
        variant="contained"
        color="primary"
        startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <DownloadIcon />}
        disabled={loading}
        onClick={handleDownload}
        fullWidth
        sx={{
          py: 1.5,
          borderRadius: 2,
          textTransform: 'none',
          fontWeight: 600,
        }}
      >
        {loading ? 'Generating PDF...' : 'Download Hierarchical Report'}
      </Button>
      
      {/* Options Dialog */}
      <Dialog open={showOptions} onClose={handleCloseOptions}>
        <DialogTitle>Hierarchical PDF Options</DialogTitle>
        <DialogContent>
          <Box sx={{ minWidth: 300, pt: 1 }}>
            <Typography variant="body2" color="text.secondary" paragraph>
              Choose which content to include in your hierarchical model PDF report:
            </Typography>
            
            <Divider sx={{ my: 2 }} />
            
            <FormControlLabel
              control={
                <Checkbox
                  checked={includeVisualization}
                  onChange={(e) => setIncludeVisualization(e.target.checked)}
                />
              }
              label="Include Pillar Visualizations"
            />
            
            <FormControlLabel
              control={
                <Checkbox
                  checked={includeRecommendations}
                  onChange={(e) => setIncludeRecommendations(e.target.checked)}
                />
              }
              label="Include Pillar Recommendations"
            />
            
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              The report will include hierarchical model results and traditional analysis.
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
          Hierarchical PDF generated successfully!
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

export default HierarchicalPDFButton; 