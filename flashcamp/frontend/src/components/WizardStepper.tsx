import React from "react";
import Stepper from "@mui/material/Stepper";
import Step from "@mui/material/Step";
import StepLabel from "@mui/material/StepLabel";

interface WizardStepperProps {
  steps: string[];
  activeStep: number;
  onJump?: (idx: number) => void;
}

const WizardStepper: React.FC<WizardStepperProps> = ({ steps, activeStep, onJump }) => (
  <Stepper orientation="vertical" activeStep={activeStep} sx={{ background: 'transparent' }}>
    {steps.map((label, idx) => (
      <Step key={label} onClick={() => onJump && onJump(idx)}>
        <StepLabel
          style={{
            cursor: onJump ? 'pointer' : 'default',
            fontWeight: activeStep === idx ? 800 : 500,
            color: activeStep === idx ? '#3B82F6' : '#222B45',
            fontSize: activeStep === idx ? 18 : 16,
            transition: 'all 0.2s',
          }}
          icon={activeStep === idx ? (
            <span style={{
              display: 'inline-block',
              background: '#3B82F6',
              color: '#fff',
              borderRadius: '50%',
              width: 28,
              height: 28,
              lineHeight: '28px',
              textAlign: 'center',
              fontWeight: 700,
              boxShadow: '0 2px 8px rgba(59,130,246,0.18)'
            }}>{idx + 1}</span>
          ) : (idx + 1)}
        >
          {label}
        </StepLabel>
      </Step>
    ))}
  </Stepper>
);

export default WizardStepper;
