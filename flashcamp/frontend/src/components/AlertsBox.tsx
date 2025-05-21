import React from "react";
import { 
  Box, 
  Typography, 
  Paper, 
  alpha, 
  Collapse
} from "@mui/material";
import { 
  Warning as WarningIcon, 
  Info as InfoIcon,
  Error as ErrorIcon
} from "@mui/icons-material";
import { motion } from "framer-motion";

interface Alert {
  type: string;
  message: string;
  severity: string;
}

interface AlertsBoxProps {
  alerts: Alert[];
}

const COLOR_PALETTE = {
  warning: "#F59E0B",
  warningLight: "#FEF3C7",
  warningBorder: "#F59E42",
  info: "#3B82F6",
  infoLight: "#E0F2FE",
  infoBorder: "#60A5FA",
  error: "#EF4444",
  errorLight: "#FEE2E2",
  errorBorder: "#F87171",
  text: "#111827"
};

const AlertsBox: React.FC<AlertsBoxProps> = ({ alerts }) => {
  if (!alerts || alerts.length === 0) return null;
  
  const getAlertColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'warning':
        return {
          bg: COLOR_PALETTE.warningLight,
          border: COLOR_PALETTE.warningBorder,
          icon: <WarningIcon sx={{ color: COLOR_PALETTE.warning, mt: 0.3, fontSize: 22 }} />
        };
      case 'error':
        return {
          bg: COLOR_PALETTE.errorLight,
          border: COLOR_PALETTE.errorBorder,
          icon: <ErrorIcon sx={{ color: COLOR_PALETTE.error, mt: 0.3, fontSize: 22 }} />
        };
      case 'info':
      default:
        return {
          bg: COLOR_PALETTE.infoLight,
          border: COLOR_PALETTE.infoBorder,
          icon: <InfoIcon sx={{ color: COLOR_PALETTE.info, mt: 0.3, fontSize: 22 }} />
        };
    }
  };
  
  return (
    <Box sx={{ width: '100%' }}>
      <Typography 
        variant="h6" 
        component="h3" 
        sx={{ 
          mb: 2, 
          fontWeight: 600, 
          color: '#111827',
          display: 'flex',
          alignItems: 'center',
          gap: 1
        }}
      >
        <WarningIcon sx={{ color: COLOR_PALETTE.warning }} />
        Alerts & Recommendations
      </Typography>
      
      <Box sx={{
        display: 'flex',
        flexDirection: 'column',
        gap: 2,
        width: '100%'
      }}>
        {alerts.map((alert, idx) => {
          const alertStyle = getAlertColor(alert.severity);
          
          return (
          <Collapse key={idx} in={true} timeout={300}>
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: idx * 0.1 }}
            >
              <Paper 
                elevation={0}
                sx={{
                    bgcolor: alertStyle.bg,
                  color: COLOR_PALETTE.text,
                    border: `1px solid ${alertStyle.border}`,
                  borderRadius: 2,
                  px: 3,
                  py: 2.5,
                  fontWeight: 500,
                  fontSize: 15,
                    boxShadow: `0 2px 10px ${alpha(alertStyle.border, 0.1)}`,
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 1.5,
                  transition: 'all 0.2s ease',
                  '&:hover': {
                      boxShadow: `0 4px 12px ${alpha(alertStyle.border, 0.15)}`,
                    transform: 'translateY(-2px)'
                  }
                }}
              >
                  {alertStyle.icon}
                <Typography 
                  variant="body1" 
                  component="div"
                  sx={{ 
                    fontWeight: 500,
                    lineHeight: 1.5
                  }}
                >
                    {alert.message}
                </Typography>
              </Paper>
            </motion.div>
          </Collapse>
          );
        })}
      </Box>
    </Box>
  );
};

export default AlertsBox;
