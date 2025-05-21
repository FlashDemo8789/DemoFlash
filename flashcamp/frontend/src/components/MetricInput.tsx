import React from "react";
import { 
  TextField, 
  Slider, 
  Switch, 
  Select, 
  MenuItem, 
  Tooltip, 
  InputLabel, 
  FormControl, 
  Box, 
  Typography,
  alpha,
  IconButton,
  FormHelperText,
  InputAdornment
} from "@mui/material";
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';

// Define local types since imports might not be available
// These should match the types in ../types/metrics.ts
interface Metric {
  name: string;
  label: string;
  pillar: string;
  type: string;
  tip?: string;
  options?: string[];
  min?: number;
  max?: number;
  step?: number;
  [key: string]: any;
}

// Enum duplicating the one in metrics.ts to avoid import issues
enum MetricType {
  number = "number",
  checkbox = "checkbox",
  text = "text",
  list = "list",
  select = "select",
  slider = "slider",
  switch = "switch"
}

interface MetricInputProps {
  metric: Metric;
  value: any;
  onChange: (value: any) => void;
}

// Modern color palette
const COLOR_PALETTE = {
  primary: "#5E60CE",
  border: "#E5E7EB",
  background: "#F9FAFB",
  text: "#111827",
  textLight: "#6B7280",
  focus: "#4F46E5"
};

const MetricInput: React.FC<MetricInputProps> = ({ metric, value, onChange }) => {
  const { name, label, type, tip, options, min, max, step } = metric;

  // Helper to format values for display
  const getDisplayValue = () => {
    if (value === null || value === undefined) return '';
    if (type === "list" && Array.isArray(value)) return value.join(', ');
    return value;
  };

  // Helper to ensure null values for empty inputs across all types
  const ensureProperValue = (value: any, type: string): any => {
    if (value === '' || value === undefined || value === null) {
      return null;
    }
    
    switch (type) {
      case 'number':
        const num = Number(value);
        // Special handling for net_profit_margin_percent
        if (name === "net_profit_margin_percent" && num < 0) {
          return 0; // Ensure non-negative value for net_profit_margin_percent
        }
        return isNaN(num) ? null : num;
      case 'checkbox':
      case 'switch':
        return Boolean(value);
      case 'list':
        if (typeof value === 'string') {
          const items = value.split(',').map(item => item.trim()).filter(Boolean);
          return items.length > 0 ? items : null;
        }
        if (Array.isArray(value)) {
          return value.length > 0 ? value : null;
        }
        return null;
      default:
        return typeof value === 'string' ? value.trim() : value;
    }
  };

  let input = null;
  switch (type) {
    case "slider":
      input = (
        <Box sx={{ px: 1, pt: 1, pb: 0.5 }}>
          <Slider
            value={typeof value === 'number' ? value : min || 0}
            min={min}
            max={max}
            step={step || 1}
            onChange={(_, val) => {
              // Ensure we always pass a number, never undefined
              const numValue = Array.isArray(val) ? val[0] : val;
              onChange(ensureProperValue(numValue, 'number'));
            }}
            valueLabelDisplay="auto"
            sx={{
              color: COLOR_PALETTE.primary,
              '& .MuiSlider-thumb': { 
                width: 14,
                height: 14,
                boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
                '&:before': {
                  boxShadow: '0 0 0 8px rgba(94, 96, 206, 0.1)',
                },
                '&:hover, &.Mui-focusVisible': {
                  boxShadow: '0 0 0 8px rgba(94, 96, 206, 0.16)',
                },
              },
              '& .MuiSlider-rail': { 
                opacity: 0.3,
                backgroundColor: alpha(COLOR_PALETTE.primary, 0.3),
              },
              '& .MuiSlider-mark': { 
                backgroundColor: COLOR_PALETTE.primary,
                height: 8,
                width: 1,
                marginTop: -3
              },
            }}
          />
        </Box>
      );
      break;
    case "switch":
      input = (
        <Box sx={{ mt: 0.5 }}>
          <Switch
            checked={Boolean(value)}
            onChange={e => {
              // Always pass a true boolean value
              onChange(ensureProperValue(e.target.checked, 'checkbox'));
            }}
            sx={{
              '& .MuiSwitch-switchBase.Mui-checked': {
                color: COLOR_PALETTE.primary,
                '&:hover': {
                  backgroundColor: alpha(COLOR_PALETTE.primary, 0.1),
                },
              },
              '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                backgroundColor: COLOR_PALETTE.primary,
              },
              '& .MuiSwitch-track': {
                borderRadius: 22 / 2,
              }
            }}
          />
        </Box>
      );
      break;
    case "select":
      input = (
        <FormControl fullWidth variant="outlined" size="small">
          <Select
            value={value || ''}
            onChange={e => {
              // Handle empty string as null
              onChange(e.target.value === '' ? null : e.target.value);
            }}
            displayEmpty
            sx={{
              borderRadius: 1.5,
              backgroundColor: COLOR_PALETTE.background,
              fontSize: 15,
              '& .MuiOutlinedInput-notchedOutline': {
                borderColor: alpha(COLOR_PALETTE.border, 0.8),
              },
              '&:hover .MuiOutlinedInput-notchedOutline': {
                borderColor: alpha(COLOR_PALETTE.primary, 0.5),
              },
              '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                borderColor: COLOR_PALETTE.primary,
              }
            }}
            MenuProps={{
              PaperProps: {
                sx: {
                  borderRadius: 2,
                  boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
                  mt: 0.5
                }
              }
            }}
          >
            <MenuItem value="" disabled>
              <Typography sx={{ color: COLOR_PALETTE.textLight, fontSize: 14 }}>Select an option</Typography>
            </MenuItem>
            {options && options.map((opt: string) => (
              <MenuItem 
                key={opt} 
                value={opt}
                sx={{ 
                  fontSize: 14, 
                  py: 1,
                  '&.Mui-selected': {
                    backgroundColor: alpha(COLOR_PALETTE.primary, 0.08),
                    '&:hover': {
                      backgroundColor: alpha(COLOR_PALETTE.primary, 0.12),
                    }
                  }
                }}
              >
                {opt}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      );
      break;
    case "number":
      input = (
        <TextField
          type="number"
          fullWidth
          size="small"
          variant="outlined"
          value={value === null || value === undefined ? '' : value}
          onChange={e => {
            // Convert to proper number or null for backend
            onChange(ensureProperValue(e.target.value, 'number'));
          }}
          // For net_profit_margin_percent field, enforce min value of 0
          InputProps={{
            sx: {
              borderRadius: 1.5,
              backgroundColor: COLOR_PALETTE.background,
              fontSize: 15
            },
            inputProps: name === "net_profit_margin_percent" ? { min: 0 } : {} // Add HTML5 validation
          }}
          sx={{
            '& .MuiOutlinedInput-root': {
              '& fieldset': {
                borderColor: alpha(COLOR_PALETTE.border, 0.8),
              },
              '&:hover fieldset': {
                borderColor: alpha(COLOR_PALETTE.primary, 0.5),
              },
              '&.Mui-focused fieldset': {
                borderColor: COLOR_PALETTE.primary,
              },
            }
          }}
        />
      );
      break;
    case "checkbox":
      input = (
        <Box sx={{ mt: 0.5 }}>
          <Switch
            checked={Boolean(value)}
            onChange={e => {
              // Always pass a true boolean value
              onChange(ensureProperValue(e.target.checked, 'checkbox'));
            }}
            sx={{
              '& .MuiSwitch-switchBase.Mui-checked': {
                color: COLOR_PALETTE.primary,
                '&:hover': {
                  backgroundColor: alpha(COLOR_PALETTE.primary, 0.1),
                },
              },
              '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                backgroundColor: COLOR_PALETTE.primary,
              },
              '& .MuiSwitch-track': {
                borderRadius: 22 / 2,
              }
            }}
          />
        </Box>
      );
      break;
    case "list":
      // Special handling for list type
      input = (
        <TextField
          fullWidth
          size="small"
          variant="outlined"
          value={Array.isArray(value) ? value.join(', ') : (value || '')}
          onChange={e => {
            // Convert comma-separated string to array for backend
            onChange(ensureProperValue(e.target.value, 'list'));
          }}
          placeholder={`Enter ${label.toLowerCase()} (comma-separated)`}
          InputProps={{
            sx: {
              borderRadius: 1.5,
              backgroundColor: COLOR_PALETTE.background,
              fontSize: 15
            },
            endAdornment: (
              <InputAdornment position="end">
                <Typography 
                  variant="caption" 
                  color="textSecondary" 
                  sx={{ fontSize: 11, opacity: 0.7 }}
                >
                  Comma-separated
                </Typography>
              </InputAdornment>
            )
          }}
          sx={{
            '& .MuiOutlinedInput-root': {
              '& fieldset': {
                borderColor: alpha(COLOR_PALETTE.border, 0.8),
              },
              '&:hover fieldset': {
                borderColor: alpha(COLOR_PALETTE.primary, 0.5),
              },
              '&.Mui-focused fieldset': {
                borderColor: COLOR_PALETTE.primary,
              },
            }
          }}
        />
      );
      break;
    default:
      // Text and other types
      input = (
        <TextField
          fullWidth
          size="small"
          variant="outlined"
          value={value || ''}
          onChange={e => {
            // For text fields, pass empty values as null to backend
            onChange(ensureProperValue(e.target.value, 'text'));
          }}
          placeholder={`Enter ${label.toLowerCase()}`}
          InputProps={{
            sx: {
              borderRadius: 1.5,
              backgroundColor: COLOR_PALETTE.background,
              fontSize: 15
            }
          }}
          sx={{
            '& .MuiOutlinedInput-root': {
              '& fieldset': {
                borderColor: alpha(COLOR_PALETTE.border, 0.8),
              },
              '&:hover fieldset': {
                borderColor: alpha(COLOR_PALETTE.primary, 0.5),
              },
              '&.Mui-focused fieldset': {
                borderColor: COLOR_PALETTE.primary,
              },
            }
          }}
        />
      );
      break;
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Typography 
            sx={{ 
              fontWeight: 600, 
              fontSize: 15, 
              color: COLOR_PALETTE.text
            }}
          >
            {label}
          </Typography>
          {tip && (
            <Tooltip 
              title={tip} 
              arrow 
              placement="top"
              componentsProps={{
                tooltip: {
                  sx: {
                    bgcolor: COLOR_PALETTE.text,
                    '& .MuiTooltip-arrow': {
                      color: COLOR_PALETTE.text,
                    },
                    borderRadius: 1.5,
                    p: 1.5,
                    maxWidth: 300,
                    boxShadow: '0 4px 20px rgba(0,0,0,0.2)',
                    fontSize: 13
                  }
                }
              }}
            >
              <IconButton 
                size="small" 
                sx={{ 
                  ml: 0.5, 
                  color: alpha(COLOR_PALETTE.text, 0.6),
                  '&:hover': {
                    backgroundColor: alpha(COLOR_PALETTE.primary, 0.1),
                    color: COLOR_PALETTE.primary
                  },
                  padding: '2px'
                }}
              >
                <HelpOutlineIcon sx={{ fontSize: 16 }} />
              </IconButton>
            </Tooltip>
          )}
        </Box>
      </Box>
      
      {input}
      
      {tip && type !== "slider" && type !== "switch" && (
        <FormHelperText sx={{ 
          ml: 0, 
          mt: 0.5, 
          fontSize: 12, 
          color: alpha(COLOR_PALETTE.textLight, 0.8),
          lineHeight: 1.3
        }}>
          {tip}
        </FormHelperText>
      )}
    </Box>
  );
};

export default MetricInput;
