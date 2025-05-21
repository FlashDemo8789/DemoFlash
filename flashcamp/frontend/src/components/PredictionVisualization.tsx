import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  CircularProgress,
  Button,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tooltip,
  Paper
} from '@mui/material';
import { styled } from '@mui/material/styles';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import ZoomOutIcon from '@mui/icons-material/ZoomOut';
import DownloadIcon from '@mui/icons-material/Download';
import FullscreenIcon from '@mui/icons-material/Fullscreen';
import CloudDownloadIcon from '@mui/icons-material/CloudDownload';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import { getVisualizationBlob, getVisualizationUrl } from '../services/hierarchicalModelService';

// Define props interface
interface PredictionVisualizationProps {
  startupData: Record<string, any>;
  isLoading?: boolean;
  error?: string | null;
}

// Styled components
const VisualizationContainer = styled(Box)(({ theme }) => ({
  position: 'relative',
  width: '100%',
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  minHeight: 400,
  backgroundColor: theme.palette.grey[50],
  borderRadius: theme.shape.borderRadius,
  padding: theme.spacing(2),
  overflow: 'hidden'
}));

const ControlsContainer = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: theme.spacing(1),
  right: theme.spacing(1),
  backgroundColor: 'rgba(255, 255, 255, 0.8)',
  borderRadius: theme.shape.borderRadius,
  padding: theme.spacing(0.5),
  display: 'flex',
  zIndex: 10
}));

const StyledImage = styled('img')<{ zoomLevel: number }>(({ zoomLevel }) => ({
  maxWidth: '100%',
  transform: `scale(${zoomLevel})`,
  transition: 'transform 0.3s ease',
  transformOrigin: 'center center'
}));

// Component implementation
const PredictionVisualization: React.FC<PredictionVisualizationProps> = ({
  startupData,
  isLoading = true,
  error = null
}) => {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(isLoading);
  const [errorState, setErrorState] = useState<string | null>(error);
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const [fullscreen, setFullscreen] = useState<boolean>(false);
  
  useEffect(() => {
    const fetchVisualization = async () => {
      setLoading(true);
      setErrorState(null);
      
      try {
        console.log('Fetching visualization for startup data:', startupData);
        
        // Filter out non-numeric data and metadata
        const filteredData = Object.entries(startupData)
          .filter(([key, value]) => 
            typeof value === 'number' && 
            !['capital_score', 'advantage_score', 'market_score', 'people_score', 'overall_score', 'success_probability'].includes(key)
          )
          .reduce((obj, [key, value]) => ({ ...obj, [key]: value }), {});
        
        // Add required fields
        const metricsData = {
          startup_id: startupData.startup_id || "demo",
          startup_name: startupData.startup_name || "Demo Startup",
          sector: startupData.sector || "Technology",
          ...filteredData
        };
        
        console.log('Sending filtered metrics data for visualization:', metricsData);
        
        // Get visualization as blob
        const blob = await getVisualizationBlob(metricsData);
        const url = URL.createObjectURL(blob);
        setImageUrl(url);
        console.log('Successfully loaded visualization image');
      } catch (err) {
        console.error('Error fetching visualization:', err);
        setErrorState(err instanceof Error ? err.message : 'Failed to load visualization');
      } finally {
        setLoading(false);
      }
    };
    
    if (startupData && Object.keys(startupData).length > 0) {
      fetchVisualization();
    } else {
      setErrorState('No startup data provided');
      setLoading(false);
    }
    
    // Clean up Blob URL on unmount
    return () => {
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
    };
  }, [startupData]);
  
  // Handle zooming
  const handleZoomIn = () => {
    setZoomLevel(prev => Math.min(prev + 0.1, 2.0));
  };
  
  const handleZoomOut = () => {
    setZoomLevel(prev => Math.max(prev - 0.1, 0.5));
  };
  
  // Handle download
  const handleDownload = async () => {
    if (imageUrl) {
      const link = document.createElement('a');
      link.href = imageUrl;
      link.download = `startup-prediction-${new Date().toISOString().slice(0, 10)}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } else {
      try {
        // If no URL is cached, fetch it directly
        const blob = await getVisualizationBlob(startupData);
        const url = URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = `startup-prediction-${new Date().toISOString().slice(0, 10)}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        // Clean up
        URL.revokeObjectURL(url);
      } catch (err) {
        console.error('Error downloading visualization:', err);
        setErrorState(err instanceof Error ? err.message : 'Failed to download visualization');
      }
    }
  };
  
  // Handle fullscreen toggle
  const toggleFullscreen = () => {
    setFullscreen(!fullscreen);
    
    // Reset zoom level when entering/exiting fullscreen
    setZoomLevel(1);
  };
  
  // Retry loading the visualization
  const handleRetry = () => {
    setLoading(true);
  };
  
  // Render loading state
  const renderLoading = () => (
    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
      <CircularProgress size={40} />
      <Typography variant="body2" color="text.secondary">
        Generating visualization...
      </Typography>
    </Box>
  );
  
  // Render error state
  const renderError = () => (
    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
      <ErrorOutlineIcon color="error" sx={{ fontSize: 48 }} />
      <Typography variant="body2" color="error">
        {errorState}
      </Typography>
      <Button 
        variant="outlined" 
        onClick={handleRetry}
        startIcon={<CloudDownloadIcon />}
      >
        Try Again
      </Button>
    </Box>
  );
  
  // Render visualization
  const renderVisualization = () => (
    <>
      <ControlsContainer>
        <Tooltip title="Zoom in">
          <IconButton size="small" onClick={handleZoomIn}>
            <ZoomInIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        <Tooltip title="Zoom out">
          <IconButton size="small" onClick={handleZoomOut}>
            <ZoomOutIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        <Tooltip title="Download">
          <IconButton size="small" onClick={handleDownload}>
            <DownloadIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        <Tooltip title="Fullscreen">
          <IconButton size="small" onClick={toggleFullscreen}>
            <FullscreenIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </ControlsContainer>
      
      <StyledImage 
        src={imageUrl || ''} 
        alt="Startup Prediction Visualization" 
        zoomLevel={zoomLevel}
      />
    </>
  );
  
  // Main component render
  return (
    <>
      <Card elevation={3}>
        <CardContent>
          <Typography variant="h5" component="h2" gutterBottom>
            Prediction Visualization
          </Typography>
          
          <VisualizationContainer>
            {loading ? renderLoading() : errorState ? renderError() : renderVisualization()}
          </VisualizationContainer>
          
          <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              Visualization based on {Object.keys(startupData).length} startup metrics
            </Typography>
            
            <Button 
              variant="outlined" 
              startIcon={<DownloadIcon />}
              onClick={handleDownload}
              disabled={!imageUrl}
            >
              Download Image
            </Button>
          </Box>
        </CardContent>
      </Card>
      
      {/* Fullscreen Dialog */}
      <Dialog
        open={fullscreen}
        onClose={toggleFullscreen}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Prediction Visualization
          <Typography variant="body2" color="text.secondary">
            {startupData.sector ? `Sector: ${startupData.sector}` : 'Detailed View'}
          </Typography>
        </DialogTitle>
        
        <DialogContent dividers>
          <Box sx={{ position: 'relative', minHeight: 500, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <ControlsContainer>
              <Tooltip title="Zoom in">
                <IconButton size="small" onClick={handleZoomIn}>
                  <ZoomInIcon fontSize="small" />
                </IconButton>
              </Tooltip>
              <Tooltip title="Zoom out">
                <IconButton size="small" onClick={handleZoomOut}>
                  <ZoomOutIcon fontSize="small" />
                </IconButton>
              </Tooltip>
              <Tooltip title="Download">
                <IconButton size="small" onClick={handleDownload}>
                  <DownloadIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </ControlsContainer>
            
            {imageUrl && (
              <img 
                src={imageUrl} 
                alt="Startup Prediction Visualization" 
                style={{ 
                  maxWidth: '100%', 
                  maxHeight: '70vh',
                  transform: `scale(${zoomLevel})`,
                  transition: 'transform 0.3s ease'
                }} 
              />
            )}
          </Box>
        </DialogContent>
        
        <DialogActions>
          <Button onClick={toggleFullscreen}>Close</Button>
          <Button 
            variant="contained" 
            startIcon={<DownloadIcon />}
            onClick={handleDownload}
            disabled={!imageUrl}
          >
            Download
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default PredictionVisualization; 