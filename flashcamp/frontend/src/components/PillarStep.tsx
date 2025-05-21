import React from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import MetricInput from "./MetricInput";

// Define interfaces that mirror those in the types file
interface Metric {
  name: string;
  label: string;
  pillar: string;
  type: string;
  tip?: string;
  options?: string[];
  [key: string]: any;
}

interface Pillar {
  label: string;
  weight: number;
  metrics: Metric[];
}

interface FormValues {
  [key: string]: any;
}

interface PillarStepProps {
  pillar: Pillar;
  values: FormValues;
  onChange: (name: string, value: any) => void;
}

const PillarStep: React.FC<PillarStepProps> = ({ pillar, values, onChange }) => (
  <Box>
    <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
      <Typography variant="h4" sx={{ fontWeight: 800, mr: 2, color: 'primary.main', letterSpacing: 1 }}>{pillar.label}</Typography>
      <Box sx={{ bgcolor: 'primary.main', color: 'white', px: 2, py: 0.5, borderRadius: 2, fontWeight: 700, fontSize: 18, boxShadow: 2 }}>
        {pillar.weight}%
      </Box>
    </Box>
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 3 }}>
      {pillar.metrics.map(metric => (
        <MetricInput
          key={metric.name}
          metric={metric}
          value={values[metric.name]}
          onChange={val => onChange(metric.name, val)}
        />
      ))}
    </Box>
  </Box>
);

export default PillarStep;
