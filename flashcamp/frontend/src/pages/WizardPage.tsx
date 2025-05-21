import React, { useState, useEffect, Suspense } from "react";
import { 
  Box,
  Button,
  Typography,
  Container,
  Stack,
  CircularProgress,
  useMediaQuery,
  useTheme,
  Paper,
  Avatar,
  IconButton,
  TextField,
  Divider,
  alpha,
  Chip,
  Tooltip,
  LinearProgress
} from "@mui/material";
import { 
  ArrowForward as ArrowForwardIcon,
  ArrowBack as ArrowBackIcon,
  Business as BusinessIcon,
  Bolt as BoltIcon,
  TrendingUp as TrendingUpIcon,
  Lightbulb as LightbulbIcon,
  Public as PublicIcon,
  People as PeopleIcon,
  Info as InfoIcon,
  InsertChartOutlined as ChartIcon,
  Refresh as RefreshIcon,
  Analytics as AnalyticsIcon
} from "@mui/icons-material";
import { motion, AnimatePresence } from "framer-motion";
import MetricInput from "../components/MetricInput";
import metricsJsonRaw from "../constants/metrics.json";
import sampleData from "../constants/sampleData.json";
import { v4 as uuidv4 } from 'uuid';
import confetti from 'canvas-confetti';
import { Metric, MetricPillar, MetricType, MetricsByPillar } from "../../types/metrics";

// Add improved interfaces that properly handle null values
interface FormState {
  [key: string]: string | number | boolean | string[] | null | undefined;
  startup_name: string;
  startup_id: string;
}

// Backend submission interface with explicit any to accommodate all metric types
interface BackendSubmissionData {
  [key: string]: any; // Using any here to avoid TypeScript errors with property assignment
  startup_name: string;
  startup_id: string;
}

// Types
interface StepData {
  key: string;
  label: string;
  metrics: Metric[];
  icon: React.ReactNode;
  weight?: number;
  description?: string;
  color: string;
}

// Modern color palette
const COLOR_PALETTE = {
  background: "#f7f9fc",
  primary: "#5E60CE",
  secondary: "#64DFDF",
  accent1: "#48BFE3",
  accent2: "#80FFDB",
  accent3: "#7400B8",
  text: "#111827",
  textLight: "#6B7280",
  success: "#10B981",
  error: "#EF4444",
  warning: "#F59E0B",
  info: "#3B82F6",
  surface: "#FFFFFF",
  border: "#E5E7EB"
};

// Define pillar keys for consistent mapping between enum and internal keys
type PillarKey = "info" | "capital" | "advantage" | "market" | "people";
const mapPillarKeyToEnum: Record<PillarKey, MetricPillar> = {
  info: MetricPillar.Info,
  capital: MetricPillar.Capital,
  advantage: MetricPillar.Advantage,
  market: MetricPillar.Market,
  people: MetricPillar.People
};

// Define new pillar colors
const PILLAR_COLORS: Record<PillarKey, string> = {
  info: "#3B82F6",
  capital: "#10B981",
  advantage: "#8B5CF6", 
  market: "#F43F5E",
  people: "#F59E0B"
};

// Icons for each pillar
const PILLAR_ICONS: Record<PillarKey, React.ReactNode> = {
  info: <InfoIcon sx={{ fontSize: 28 }} />,
  capital: <TrendingUpIcon sx={{ fontSize: 28 }} />,
  advantage: <LightbulbIcon sx={{ fontSize: 28 }} />,
  market: <PublicIcon sx={{ fontSize: 28 }} />,
  people: <PeopleIcon sx={{ fontSize: 28 }} />
};

// Define pillar descriptions
const PILLAR_DATA: Array<StepData & { key: PillarKey }> = [
  { 
    key: "info", 
    label: "Startup Info", 
    icon: PILLAR_ICONS.info,
    color: PILLAR_COLORS.info,
    description: "Basic information about your startup",
    metrics: [] 
  },
  { 
    key: "capital", 
    label: "Capital", 
    weight: 30, 
    icon: PILLAR_ICONS.capital,
    color: PILLAR_COLORS.capital,
    description: "Funding, runway, and financial metrics",
    metrics: []
  },
  { 
    key: "advantage", 
    label: "Advantage",  
    weight: 20, 
    icon: PILLAR_ICONS.advantage,
    color: PILLAR_COLORS.advantage,
    description: "Competitive edge, IP, and market positioning",
    metrics: []
  },
  { 
    key: "market", 
    label: "Market", 
    weight: 25, 
    icon: PILLAR_ICONS.market,
    color: PILLAR_COLORS.market,
    description: "Market size, growth rate, and competition",
    metrics: []
  },
  { 
    key: "people", 
    label: "People", 
    weight: 25, 
    icon: PILLAR_ICONS.people,
    color: PILLAR_COLORS.people,
    description: "Team, expertise, and culture metrics",
    metrics: []
  }
];

// Group metrics by pillar
const groupMetricsByPillar = (metrics: Metric[]): StepData[] => {
  return PILLAR_DATA.filter(p => p.key !== "info").map(pillar => ({
    ...pillar,
    metrics: metrics.filter(m => m.pillar.toLowerCase() === pillar.key)
  }));
};

// metricsJsonRaw is the raw JSON, so cast it to Metric[]
const metricsJson: Metric[] = metricsJsonRaw as Metric[];

// Function to populate form with sample data
const populateWithSampleData = (): FormState => {
  return { ...sampleData } as unknown as FormState;
};

// Define constants for required fields based on backend schema
const REQUIRED_FIELDS = [
  'startup_id',
  'funding_stage',
  'team_size_full_time',
  'monthly_burn_usd',
  'cash_on_hand_usd',
  'market_growth_rate_percent',
  'founder_domain_experience_years'
];

// Comprehensive default values for ALL metrics to facilitate testing
const DEFAULT_VALUES: Record<string, any> = {
  // Required fields with sensible defaults
  funding_stage: 'Seed',
  team_size_full_time: 5,
  monthly_burn_usd: 50000,
  cash_on_hand_usd: 500000,
  market_growth_rate_percent: 15,
  founder_domain_experience_years: 7,
  
  // Optional fields with typical values for a promising seed-stage startup
  startup_name: 'FlashTech AI',
  startup_id: '', // Will be generated dynamically
  sector: 'Fintech',
  subsector: 'Investment Analytics',
  market_geography: ['North America', 'Europe'],
  country: 'USA',
  website: 'https://teststart.up',
  founding_year: 2023,
  funding_rounds_count: 1,
  committed_funding_usd: 500000,
  team_size_total: 8,
  technical_team_size: 4,
  founding_team_size: 2,
  team_diversity_percent: 40,
  employee_turnover_rate_percent: 5,
  team_engagement_score: 8,
  key_person_dependency: 3,
  complementary_skills_index: 0.8,
  burn_stddev_usd: 5000,
  gross_margin_percent: 70,
  net_profit_margin_percent: -20,
  annual_revenue_run_rate: 120000,
  revenue_growth_rate: 15,
  tam_size_usd: 5000000000,
  sam_size_usd: 500000000,
  claimed_tam_usd: 10000000000,
  claimed_cagr_pct: 25,
  tam_justification: 'Based on industry reports and addressable segments',
  market_adoption_stage: 'Early Adopters',
  competition_intensity: 6,
  competitors_named_count: 3,
  top3_competitor_share_pct: 45,
  nps_score: 65,
  waitlist_size: 250,
  mau: 500,
  ltv_cac_ratio: 3.5,
  customer_churn_rate_percent: 2,
  cac_payback_months: 8,
  ltv_usd: 25000,
  cac_usd: 7000,
  customer_concentration_percent: 30,
  product_retention_30d: 80,
  product_retention_90d: 65,
  conversion_rate_percent: 2.5,
  user_growth_rate_percent: 12,
  patents_count: 1,
  has_patents: true,
  has_data_moat: true,
  tech_differentiation_score: 8,
  regulatory_advantage_present: false,
  network_effects_present: true,
  has_network_effect: true,
  switching_cost_score: 7,
  switching_cost: 'High integration cost and data migration complexity',
  product_stage: 'Beta',
  product_uptime_percent: 99.5,
  prior_startup_experience_count: 1,
  prior_successful_exits_count: 0,
  previous_exit_max_value_usd: 0,
  previous_exits_count: 0,
  board_advisor_experience_score: 8,
  founders_count: 2,
  years_experience_avg: 10,
  degree_level_mode: 'Masters',
  gender_diversity_index: 0.5,
  geography_diversity_index: 0.3,
  linkedin_connections_total: 3500,
  twitter_followers_total: 5000,
  has_debt: true,
  sales_cycle_length_days: 45,
  scalability_score: 7,
  distribution_partner_count: 2,
  brand_strength_score: 6,
  viral_coefficient: 0.8,
  regulatory_risk_score: 4,
  industry_regulation_level: 'Medium',
  website_health_score: 85,
  data_source: 'Manual',
  success_label: null,
  
  // Additional metrics to ensure 100% coverage
  industry: 'Financial Services',
  total_funding_usd: 2000000,
  product_market_fit_score: 7,
  intellectual_property_score: 6,
  market_share_percent: 0.5,
  customer_acquisition_channels: ['Direct Sales', 'Content Marketing', 'Partnerships'],
  customer_segments: ['SMB', 'Enterprise'],
  platform_integrations_count: 5,
  api_calls_monthly: 50000,
  data_processed_gb: 500,
  infrastructure_cost_monthly: 5000,
  security_audit_score: 85,
  compliance_rating: 'SOC2',
  international_presence: 2,
  localization_languages: ['English', 'Spanish'],
  mobile_adoption_percent: 60,
  web_adoption_percent: 80,
  feature_usage_score: 7.5,
  feature_release_frequency_days: 14,
  technical_debt_score: 3,
  testing_coverage_percent: 75,
  deployment_frequency_weekly: 3,
  incident_response_time_hours: 4,
  active_development_streams: 4,
  open_issues_count: 45,
  developer_satisfaction_score: 8,
  documentation_completeness: 70,
  partner_ecosystem_size: 8,
  customer_support_response_time_hours: 4,
  customer_success_team_size: 2
};

const WizardPage: React.FC = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const isTablet = useMediaQuery(theme.breakpoints.down('md'));
  
  // Info metrics (excluding startup_id and success_label)
  const INFO_METRICS: Metric[] = metricsJson.filter(m => {
    // Convert enum value to lowercase for comparison
    const pillarValue = m.pillar.toString().toLowerCase();
    return pillarValue === "info" && m.name !== "startup_id" && m.name !== "success_label";
  });
  
  // Create all steps with metrics
  const ALL_STEPS: StepData[] = [
    { ...PILLAR_DATA[0], metrics: INFO_METRICS },
    ...groupMetricsByPillar(metricsJson)
  ];

  // State variables
  const [step, setStep] = useState<number>(0);
  const [form, setForm] = useState<FormState>(() => {
    // Start with a complete set of default values
    const initialForm: FormState = {
      startup_id: uuidv4(), // Always generate a new UUID for a new startup
      startup_name: DEFAULT_VALUES.startup_name || "New Startup", // Default startup name
    };
    
    // Add default values for all metrics to facilitate testing
    metricsJson.forEach(metric => {
      const defaultValue = DEFAULT_VALUES[metric.name];
      if (defaultValue !== undefined) {
        initialForm[metric.name] = defaultValue;
      }
    });
    
    return initialForm;
  });
  const [steps, setSteps] = useState<StepData[]>([]);
  const [analyzing, setAnalyzing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [progress, setProgress] = useState<number>(0);
  const [completedSteps, setCompletedSteps] = useState<Record<number, { completed: boolean; progress: number }>>({});
  const [showConfetti, setShowConfetti] = useState<boolean>(false);

  // Track form completion status
  useEffect(() => {
    const trackCompletion = () => {
      const newCompletedSteps: Record<number, { completed: boolean; progress: number }> = {};
      ALL_STEPS.forEach((stepData, index) => {
        const totalFields = stepData.metrics.length;
        const filledFields = stepData.metrics.filter(metric => 
          form[metric.name as keyof FormState] !== undefined && form[metric.name as keyof FormState] !== ""
        ).length;
        newCompletedSteps[index] = {
          completed: filledFields === totalFields && totalFields > 0,
          progress: totalFields > 0 ? (filledFields / totalFields) * 100 : 0
        };
      });
      setCompletedSteps(newCompletedSteps);
      
      // Calculate overall progress
      const totalMetrics = ALL_STEPS.reduce((acc, s) => acc + s.metrics.length, 0);
      const filledMetrics = Object.keys(form).filter(key => form[key as keyof FormState] !== undefined && form[key as keyof FormState] !== "").length;
      setProgress((filledMetrics / totalMetrics) * 100);
    };
    trackCompletion();
  }, [form]);

  // Initialize steps and startup ID
  useEffect(() => {
    setSteps(ALL_STEPS);
  }, []);

  // Trigger confetti on successful analysis
  const triggerConfetti = (): void => {
    setShowConfetti(true);
    confetti({
      particleCount: 200,
      spread: 100,
      origin: { y: 0.3, x: 0.5 },
      colors: [PILLAR_COLORS.capital, PILLAR_COLORS.advantage, PILLAR_COLORS.market, PILLAR_COLORS.people, PILLAR_COLORS.info],
      gravity: 0.5,
      scalar: 1.2,
      shapes: ['circle', 'square']
    });
    setTimeout(() => setShowConfetti(false), 3000);
  };

  // Navigation handlers
  const handleNext = (): void => {
    setStep(s => Math.min(s + 1, steps.length - 1));
  };

  const handlePrev = (): void => {
    setStep(s => Math.max(s - 1, 0));
  };

  const handleJump = (idx: number): void => {
    if (idx !== step) {
      setStep(idx);
    }
  };

  const handleMetricChange = (name: string, value: any): void => {
    setForm((prevForm: FormState) => ({ ...prevForm, [name]: value }));
  };

  // Add function to get exact required fields from backend schema
  const getBackendRequiredFields = () => {
    // These MUST match exactly what backend expects - from schemas.py
    return [
      'startup_id',
      'startup_name',
      'funding_stage',
      'team_size_full_time',
      'monthly_burn_usd',
      'cash_on_hand_usd',
      'market_growth_rate_percent',
      'founder_domain_experience_years'
    ];
  };

  // Add auto-analysis function for testing
  const runAutoAnalysis = async () => {
    console.log("Starting automatic analysis with default values...");
    setAnalyzing(true);
    setError(null);
    
    try {
      // Create a properly typed copy of the form data for submission with complete values
      const typedFormData: BackendSubmissionData = {
        startup_name: DEFAULT_VALUES.startup_name as string,
        startup_id: form.startup_id || uuidv4(),
      };
      
      // First, ensure ALL required fields are properly set with correct types
      const requiredFields = getBackendRequiredFields();
      requiredFields.forEach(field => {
        if (field === 'startup_id') {
          typedFormData[field] = form.startup_id || uuidv4();
        } else {
          let value = DEFAULT_VALUES[field];
          // Ensure correct type conversion
          if (typeof value === 'number' || field.includes('_usd') || field.includes('_percent') || 
              field.includes('_months') || field.includes('_years') || field.includes('_size')) {
            typedFormData[field] = Number(value);
          } else if (typeof value === 'boolean') {
            typedFormData[field] = Boolean(value);
          } else {
            typedFormData[field] = value;
          }
        }
      });
      
      // Process all remaining metrics to ensure 100% form completion
      metricsJson.forEach(metric => {
        // Skip fields we've already processed
        if (requiredFields.includes(metric.name)) return;
        
        let defaultValue = DEFAULT_VALUES[metric.name];
        
        // Apply type-specific processing
        if (defaultValue !== undefined) {
          switch(metric.type) {
            case 'number':
              typedFormData[metric.name] = Number(defaultValue);
              break;
            case 'checkbox':
              typedFormData[metric.name] = Boolean(defaultValue);
              break;
            case 'list':
              if (typeof defaultValue === 'string') {
                typedFormData[metric.name] = defaultValue.split(',').map(item => item.trim()).filter(Boolean);
              } else if (Array.isArray(defaultValue)) {
                typedFormData[metric.name] = defaultValue;
              } else {
                typedFormData[metric.name] = [String(defaultValue)];
              }
              break;
            default:
              typedFormData[metric.name] = defaultValue;
          }
        } else {
          // If no default value exists, use field-specific defaults
          switch(metric.type) {
            case 'number':
              typedFormData[metric.name] = 5; // Default numerical value
              break;
            case 'checkbox':
              typedFormData[metric.name] = false;
              break;
            case 'list':
              typedFormData[metric.name] = [];
              break;
            case 'text':
              typedFormData[metric.name] = `Default ${metric.label}`;
              break;
            default:
              typedFormData[metric.name] = null;
          }
        }
      });
      
      // Business logic validation and corrections
      const cash = Number(typedFormData.cash_on_hand_usd);
      const burn = Number(typedFormData.monthly_burn_usd);
      
      if (!isNaN(cash) && !isNaN(burn) && burn > 0) {
        const expectedRunway = Math.round(cash / burn);
        typedFormData.runway_months = expectedRunway;
        console.log(`Set runway to ${expectedRunway} months based on cash / burn`);
      }
      
      // Ensure team_size_consistency
      const fullTime = Number(typedFormData.team_size_full_time);
      if (typedFormData.team_size_total !== undefined && 
          Number(typedFormData.team_size_total) < fullTime) {
        typedFormData.team_size_total = fullTime;
      }
      
      // Remove any fields that are undefined or null in optional fields
      Object.keys(typedFormData).forEach(key => {
        if (!requiredFields.includes(key) && (typedFormData[key] === undefined || typedFormData[key] === null)) {
          delete typedFormData[key];
        }
      });
      
      console.log("Submitting complete payload (100% metrics) to backend:", typedFormData);
      
      // Count metrics to verify 100% coverage
      const metricCount = Object.keys(typedFormData).length;
      console.log(`Form contains ${metricCount} metrics (target: 99+)`);
      
      // Submit to backend
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(typedFormData),
      });

      if (response.ok) {
        const data = await response.json();
        console.log("✅ Analysis successful!", data);
        setTimeout(() => {
          setResult(data);
          triggerConfetti();
        }, 800);
      } else {
        let errorDetail = "An error occurred while analyzing the startup.";
        
        try {
          // Try to parse the error response
          const contentType = response.headers.get("content-type");
          if (contentType && contentType.includes("application/json")) {
            const errorJson = await response.json();
            console.error("Full error response:", errorJson);
            errorDetail = errorJson.detail || JSON.stringify(errorJson) || errorDetail;
          } else {
            const errorText = await response.text();
            console.error("Full error text:", errorText);
            errorDetail = errorText || errorDetail;
          }
        } catch (e) {
          console.error("Error parsing API error response:", e);
        }
        
        console.error(`❌ Error submitting form: ${response.status} ${response.statusText}`, errorDetail);
        setError(errorDetail);
      }
    } catch (err) {
      console.error("Error:", err);
      setError("An unexpected error occurred. Please try again.");
    } finally {
      setAnalyzing(false);
    }
  };

  // Submit and analyze form data
  const handleAnalyze = async (): Promise<void> => {
    setAnalyzing(true);
    setError(null);
    
    try {
      // Create a properly typed copy of the form data for submission
      const typedFormData: BackendSubmissionData = {
        startup_name: form.startup_name || "New Startup",
        startup_id: form.startup_id || uuidv4(),
      };
      
      // Track missing required fields
      const missingRequiredFields: string[] = [];
      
      // First ensure all required fields have values
      for (const field of REQUIRED_FIELDS) {
        if (field === 'startup_id') continue; // Already handled above
        
        const value = form[field];
        if (value === undefined || value === null || value === '') {
          // Get default value for this field
          const defaultValue = DEFAULT_VALUES[field as keyof typeof DEFAULT_VALUES];
          if (defaultValue !== undefined) {
            typedFormData[field] = defaultValue;
          } else {
            missingRequiredFields.push(field);
          }
        } else {
          // Apply appropriate type conversion based on the field's expected type
          const metric = metricsJson.find(m => m.name === field);
          if (metric) {
            switch(metric.type) {
              case 'number':
                typedFormData[field] = Number(value);
                break;
              case 'checkbox':
                typedFormData[field] = Boolean(value);
                break;
              case 'list':
                if (typeof value === 'string') {
                  typedFormData[field] = value.split(',').map(item => item.trim()).filter(Boolean);
                } else if (Array.isArray(value)) {
                  typedFormData[field] = value;
                } else {
                  typedFormData[field] = [String(value)];
                }
                break;
              default:
                typedFormData[field] = value;
            }
          } else {
            // If we don't have metric info, just copy the value
            typedFormData[field] = value;
          }
        }
      }
      
      // Process the remaining non-required fields
      metricsJson.forEach(metric => {
        if (REQUIRED_FIELDS.includes(metric.name)) return; // Skip required fields, already processed
        
        const rawValue = form[metric.name as keyof FormState];
        let processedValue: string | number | boolean | string[] | null = null;
        
        // Apply type-specific processing to ensure backend compatibility
        if (metric.name.endsWith('_json')) {
          if (rawValue === '' || rawValue === undefined || rawValue === null) {
            processedValue = '';
          } else if (typeof rawValue === 'string') {
            processedValue = rawValue;
          } else {
            processedValue = JSON.stringify(rawValue);
          }
        } else {
          switch(metric.type) {
            case 'number':
              if (rawValue === '' || rawValue === undefined || rawValue === null) {
                processedValue = null;
              } else {
                processedValue = Number(rawValue);
              }
              break;
            case 'checkbox':
              processedValue = Boolean(rawValue);
              break;
            case 'list':
              if (rawValue === null || rawValue === undefined || rawValue === '') {
                processedValue = null;
              } else if (typeof rawValue === 'string') {
                const items = rawValue.split(',').map(item => item.trim()).filter(Boolean);
                processedValue = items.length > 0 ? items : null;
              } else if (Array.isArray(rawValue)) {
                processedValue = rawValue.length > 0 ? rawValue : null;
              } else {
                processedValue = [String(rawValue)];
              }
              break;
            case 'text':
              processedValue = (rawValue === '' || rawValue === undefined || rawValue === null) 
                ? '' 
                : String(rawValue);
              break;
            default:
              processedValue = (rawValue === '' || rawValue === undefined || rawValue === null)
                ? null
                : rawValue as string | number | boolean | string[];
          }
        }
        
        typedFormData[metric.name] = processedValue;
      });
      
      // Validate required fields
      if (missingRequiredFields.length > 0) {
        setError(`Missing required fields: ${missingRequiredFields.join(', ')}`);
        setAnalyzing(false);
        return;
      }

      // Special business logic validation based on backend model
      // Calculate runway if we have cash and burn but not manually entering it
      const cash = Number(typedFormData.cash_on_hand_usd);
      const burn = Number(typedFormData.monthly_burn_usd);
      
      if (!isNaN(cash) && !isNaN(burn) && burn > 0) {
        // We could add a calculated field for runway in months if needed
        const calculatedRunway = Math.round(cash / burn);
        console.log(`Calculated runway: ${calculatedRunway} months based on cash / burn`);
        // We don't need to set runway_months since this metric has been removed
      }
      
      // Ensure team_size consistency
      const fullTime = Number(typedFormData.team_size_full_time);
      const teamTotal = Number(typedFormData.team_size_total);
      
      if (!isNaN(fullTime) && !isNaN(teamTotal) && fullTime > teamTotal && teamTotal > 0) {
        console.log(`Fixing team_size_total from ${teamTotal} to ${fullTime} to ensure consistency with full-time`);
        typedFormData.team_size_total = fullTime;
      }
      
      console.log("Submitting typed payload to backend:", typedFormData);
      
      // Add explicit Content-Type header
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(typedFormData),
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log("Received successful response:", data);
      setTimeout(() => {
        setResult(data);
        triggerConfetti();
      }, 800);
      } else {
        let errorDetail = "An error occurred while analyzing the startup.";
        
        try {
          // Try to parse the error response
          const contentType = response.headers.get("content-type");
          if (contentType && contentType.includes("application/json")) {
            const errorJson = await response.json();
            errorDetail = errorJson.detail || errorDetail;
          } else {
            errorDetail = await response.text();
          }
        } catch (err: any) {
          console.error("Error:", err);
          errorDetail = "An unexpected error occurred. Please try again.";
          
          // Check if the error is the response object itself from a failed fetch
          if (err instanceof Error && err.message.startsWith('{') && err.message.endsWith('}') || err.message.startsWith('[') && err.message.endsWith(']')) {
            try {
              const errorJson = JSON.parse(err.message);
              // Extract detail from FastAPI's validation error structure
              if (Array.isArray(errorJson.detail)) {
                errorDetail = errorJson.detail.map((d: any) => 
                  `Field '${d.loc ? d.loc.slice(1).join('.') : 'unknown'}': ${d.msg}`
                ).join("; ");
              } else if (errorJson.detail) {
                errorDetail = errorJson.detail;
              }
            } catch (parseError) {
               // If parsing fails, use the original message if it's a string
               if (typeof err.message === 'string') {
                 errorDetail = err.message;
               }
            }
          } else if (err instanceof Error) {
               errorDetail = err.message;
          } else if (typeof err === 'string') {
             errorDetail = err;
          }
        }
        
        console.error(`Error submitting form: 422 Unprocessable Entity`, errorDetail); // Log the processed detail
        setError(errorDetail); // Set the processed string error message
      }
    } catch (err) {
      console.error("Error:", err);
      setError("An unexpected error occurred. Please try again.");
    } finally {
      setAnalyzing(false);
    }
  };

  // Add auto-analysis test button after the form
  const renderTestControls = () => {
    return (
      <Box sx={{ mt: 3, display: 'flex', justifyContent: 'center' }}>
        <Button
          variant="contained"
          color="secondary"
          onClick={runAutoAnalysis}
          disabled={analyzing}
          startIcon={<AnalyticsIcon />}
          sx={{ 
            fontWeight: 600, 
            py: 1.5, 
            px: 3,
            borderRadius: 2,
            boxShadow: '0 4px 6px rgba(0,0,0,0.12)',
            '&:hover': {
              boxShadow: '0 6px 10px rgba(0,0,0,0.18)'
            }
          }}
        >
          Run Auto-Analysis Test (100% Metrics)
        </Button>
      </Box>
    );
  };

  // Lazy load ResultsPage
  const ResultsPage = React.lazy(() => import("./ResultsPage"));

  // Loading state
  if (steps.length === 0) {
    return (
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100vh',
        flexDirection: 'column',
        background: `linear-gradient(135deg, ${alpha(COLOR_PALETTE.primary, 0.05)} 0%, ${alpha(COLOR_PALETTE.accent1, 0.1)} 100%)`
      }}>
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
        >
          <CircularProgress 
            size={60} 
            thickness={5} 
            sx={{ 
              color: COLOR_PALETTE.primary,
              boxShadow: `0 0 30px ${alpha(COLOR_PALETTE.primary, 0.2)}`
            }} 
          />
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
        >
          <Typography 
            variant="h6" 
            sx={{ 
              mt: 4, 
              fontWeight: 600, 
              color: COLOR_PALETTE.text,
              letterSpacing: '-0.01em'
            }}
          >
            Preparing your analysis...
          </Typography>
        </motion.div>
      </Box>
    );
  }

  // Results page
  if (result) {
    return (
      <Suspense fallback={
        <Box sx={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          height: '100vh',
          flexDirection: 'column',
          background: `linear-gradient(135deg, ${alpha(COLOR_PALETTE.primary, 0.05)} 0%, ${alpha(COLOR_PALETTE.accent1, 0.1)} 100%)`
        }}>
          <CircularProgress size={60} thickness={5} sx={{ color: COLOR_PALETTE.primary }} />
          <Typography variant="h6" sx={{ mt: 3, fontWeight: 600, color: COLOR_PALETTE.text }}>
            Preparing your results...
          </Typography>
        </Box>
      }>
        <ResultsPage result={result} />
      </Suspense>
    );
  }

  // Get current step data
  const currentStep = steps[step] || steps[0];
  const currentColor = currentStep.color;

  return (
    <Box sx={{
      minHeight: '100vh',
      background: `linear-gradient(135deg, ${alpha(currentColor, 0.02)} 0%, ${alpha(COLOR_PALETTE.background, 0.8)} 100%)`,
      display: 'flex',
      flexDirection: 'column',
      fontFamily: '"Inter", system-ui, -apple-system, sans-serif',
      overflow: 'hidden',
      position: 'relative',
      pt: { xs: 2, md: 3 }
    }}>
      {/* Header */}
      <Container maxWidth="xl">
        <Box sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between',
          mb: { xs: 3, md: 5 }
        }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Avatar 
              sx={{ 
                bgcolor: alpha(currentColor, 0.2), 
                color: currentColor,
                width: { xs: 38, md: 44 },
                height: { xs: 38, md: 44 },
                mr: 1.5,
                boxShadow: `0 4px 12px ${alpha(currentColor, 0.2)}`
              }}
            >
              <BoltIcon />
            </Avatar>
            <Box>
              <Typography 
                variant="h5" 
                component="h1" 
                sx={{ 
                  fontWeight: 700, 
                  color: COLOR_PALETTE.text,
                  fontSize: { xs: 20, md: 22 },
                  letterSpacing: '-0.02em'
                }}
              >
                FlashCAMP
              </Typography>
              <Typography 
                variant="body2" 
                sx={{ 
                  color: COLOR_PALETTE.textLight,
                  fontSize: 13 
                }}
              >
                Startup Analysis Wizard
              </Typography>
            </Box>
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Box sx={{ mr: 1, display: { xs: 'none', sm: 'block' } }}>
              <Typography variant="body2" sx={{ color: COLOR_PALETTE.textLight, fontWeight: 500 }}>
                Overall Progress
              </Typography>
            </Box>
            <Box sx={{ 
              position: 'relative', 
              width: { xs: 90, sm: 120, md: 150 },
              height: 6,
              borderRadius: 3,
              bgcolor: alpha(COLOR_PALETTE.border, 0.8),
              mr: 0.5
            }}>
              <Box sx={{ 
                position: 'absolute',
                top: 0,
                left: 0,
                height: '100%',
                width: `${progress}%`,
                borderRadius: 3,
                background: `linear-gradient(90deg, ${currentColor}, ${adjustColor(currentColor, 20)})`,
                transition: 'width 0.5s ease-out'
              }} />
            </Box>
            <Typography variant="caption" sx={{ fontWeight: 600, color: COLOR_PALETTE.text, ml: 0.5 }}>
              {Math.round(progress)}%
            </Typography>
          </Box>
        </Box>
      </Container>
      
      {/* Main wizard area */}
      <Container maxWidth="xl" sx={{ 
        display: 'flex', 
        flexDirection: { xs: 'column', md: 'row' }, 
        flex: 1,
        gap: { xs: 3, md: 4 },
        mb: 4
      }}>
        {/* Left panel - Steps navigation */}
        <Box sx={{ 
          width: { xs: '100%', md: '300px' },
          flexShrink: 0,
        }}>
          <Paper elevation={0} sx={{
            p: { xs: 2, md: 3 },
            borderRadius: 3,
            bgcolor: 'rgba(255, 255, 255, 0.8)',
            backdropFilter: 'blur(10px)',
            border: `1px solid ${alpha(COLOR_PALETTE.border, 0.7)}`,
            height: '100%'
          }}>
            <Typography variant="subtitle1" sx={{ 
              fontWeight: 600, 
              color: COLOR_PALETTE.text,
              mb: 2.5
            }}>
              CAMP Analysis Steps
            </Typography>
            
            <Stack spacing={1.5}>
              {steps.map((stepData, idx) => {
                const isActive = step === idx;
                const isCompleted = completedSteps[idx]?.completed;
                const stepProgress = completedSteps[idx]?.progress || 0;
                
                return (
                  <Box 
                    key={stepData.key}
                    onClick={() => handleJump(idx)}
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      p: 1.5,
                      borderRadius: 2,
                      cursor: 'pointer',
                      transition: 'all 0.2s ease',
                      background: isActive 
                        ? alpha(stepData.color, 0.1) 
                        : 'transparent',
                      '&:hover': {
                        background: isActive 
                          ? alpha(stepData.color, 0.15) 
                          : alpha(COLOR_PALETTE.border, 0.4)
                      }
                    }}
                  >
                    <Avatar 
                      sx={{
                        width: 36,
                        height: 36,
                        bgcolor: isCompleted 
                          ? COLOR_PALETTE.success 
                          : isActive 
                            ? stepData.color 
                            : alpha(COLOR_PALETTE.textLight, 0.2),
                        color: isCompleted || isActive ? '#fff' : COLOR_PALETTE.textLight,
                        mr: 1.5,
                        transition: 'all 0.3s ease'
                      }}
                    >
                      {stepData.icon}
                    </Avatar>
                    
                    <Box sx={{ flex: 1 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography sx={{ 
                          fontWeight: isActive ? 600 : 500,
                          fontSize: 15,
                          color: isActive ? stepData.color : COLOR_PALETTE.text
                        }}>
                          {stepData.label}
                        </Typography>
                        
                        {isCompleted && (
                          <Chip 
                            label="Complete" 
                            size="small" 
                            sx={{ 
                              bgcolor: alpha(COLOR_PALETTE.success, 0.1), 
                              color: COLOR_PALETTE.success,
                              height: 22,
                              fontSize: 11,
                              fontWeight: 600,
                              ml: 1
                            }} 
                          />
                        )}
                      </Box>
                      
                      {!isCompleted && (
                        <Box sx={{ 
                          mt: 0.5,
                          height: 4,
                          borderRadius: 2,
                          bgcolor: alpha(COLOR_PALETTE.border, 0.6),
                          width: '100%',
                          position: 'relative',
                          overflow: 'hidden'
                        }}>
                          <Box sx={{ 
                            position: 'absolute',
                            top: 0,
                            left: 0,
                            height: '100%',
                            width: `${stepProgress}%`,
                            borderRadius: 2,
                            bgcolor: isActive ? stepData.color : COLOR_PALETTE.textLight,
                          }} />
                        </Box>
                      )}
                    </Box>
                  </Box>
                );
              })}
            </Stack>
            
            <Box sx={{ mt: 4, p: 2, borderRadius: 2, bgcolor: alpha(PILLAR_COLORS.info, 0.08) }}>
              <Typography variant="caption" sx={{ 
                color: COLOR_PALETTE.text, 
                fontWeight: 500,
                display: 'block',
                mb: 1
              }}>
                Need assistance?
              </Typography>
              <Typography variant="body2" sx={{ color: COLOR_PALETTE.textLight, fontSize: 13 }}>
                Complete all sections to generate your comprehensive startup analysis report.
              </Typography>
            </Box>
          </Paper>
        </Box>
        
        {/* Right panel - Form content */}
        <Box sx={{ flex: 1 }}>
          <AnimatePresence mode="wait">
            <motion.div
              key={`step-${step}`}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.4 }}
              style={{ height: '100%' }}
            >
              <Paper elevation={0} sx={{
                height: '100%',
                p: { xs: 3, md: 4 },
                borderRadius: 3,
                bgcolor: 'rgba(255, 255, 255, 0.9)',
                backdropFilter: 'blur(10px)',
                border: `1px solid ${alpha(COLOR_PALETTE.border, 0.7)}`,
                position: 'relative',
                overflow: 'hidden',
                display: 'flex',
                flexDirection: 'column'
              }}>
                {/* Color strip */}
                <Box sx={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: 5,
                  background: `linear-gradient(90deg, ${currentColor}, ${adjustColor(currentColor, 20)})`
                }} />
                
                {/* Header */}
                <Box sx={{ mb: 4, display: 'flex', alignItems: 'center' }}>
                  <Avatar 
                    sx={{ 
                      bgcolor: alpha(currentColor, 0.15),
                      color: currentColor,
                      width: 52,
                      height: 52,
                      mr: 2.5,
                      boxShadow: `0 4px 12px ${alpha(currentColor, 0.2)}`
                    }}
                  >
                    {currentStep.icon}
                  </Avatar>
                  
                  <Box>
                    <Typography variant="h4" sx={{ 
                      fontWeight: 700, 
                      color: COLOR_PALETTE.text,
                      fontSize: { xs: 24, md: 28 },
                      mb: 0.5
                    }}>
                      {currentStep.label}
                    </Typography>
                    
                    <Typography variant="body1" sx={{ 
                      color: COLOR_PALETTE.textLight,
                      maxWidth: 600
                    }}>
                      {currentStep.description}
                    </Typography>
                  </Box>
                </Box>
                
                {/* Form fields */}
                <Box sx={{ 
                  flex: 1, 
                  overflowY: 'auto',
                  pr: 1,
                  pb: 2
                }}>
                  <Stack spacing={3.5}>
                    {currentStep.metrics.map((metric, idx) => (
                      <motion.div
                        key={metric.name}
                        initial={{ opacity: 0, y: 15 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.3, delay: 0.1 + (idx * 0.05) }}
                      >
                        <Paper
                          elevation={0}
                          sx={{
                            p: 2.5,
                            borderRadius: 2,
                            border: `1px solid ${alpha(COLOR_PALETTE.border, 0.7)}`,
                            transition: 'all 0.2s ease',
                            '&:hover': {
                              boxShadow: `0 4px 12px ${alpha(COLOR_PALETTE.text, 0.05)}`,
                              borderColor: alpha(currentColor, 0.3),
                            }
                          }}
                        >
                          <MetricInput
                            metric={metric}
                            value={form[metric.name as keyof FormState]}
                            onChange={val => handleMetricChange(metric.name, val)}
                          />
                        </Paper>
                      </motion.div>
                    ))}
                  </Stack>
                </Box>
                
                {/* Navigation controls */}
                <Box sx={{ 
                  display: 'flex', 
                  justifyContent: 'space-between',
                  mt: 4,
                  pt: 3,
                  borderTop: `1px solid ${alpha(COLOR_PALETTE.border, 0.7)}`
                }}>
                  <Button
                    variant="outlined"
                    disabled={step === 0}
                    onClick={handlePrev}
                    startIcon={<ArrowBackIcon />}
                    sx={{
                      borderRadius: 2,
                      py: 1.2,
                      px: 3,
                      borderColor: alpha(COLOR_PALETTE.border, 0.8),
                      color: COLOR_PALETTE.textLight,
                      fontWeight: 500,
                      '&:hover': {
                        borderColor: currentColor,
                        color: currentColor,
                        bgcolor: alpha(currentColor, 0.05)
                      },
                      visibility: step === 0 ? 'hidden' : 'visible',
                      transition: 'all 0.2s ease',
                    }}
                  >
                    Previous
                  </Button>
                  
                  {step < steps.length - 1 ? (
                    <motion.div
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <Button
                        variant="contained"
                        onClick={handleNext}
                        endIcon={<ArrowForwardIcon />}
                        disableElevation
                        sx={{
                          borderRadius: 2,
                          py: 1.2,
                          px: 3,
                          background: `linear-gradient(45deg, ${currentColor}, ${adjustColor(currentColor, 20)})`,
                          fontWeight: 600,
                          transition: 'all 0.2s ease',
                          boxShadow: `0 4px 12px ${alpha(currentColor, 0.3)}`,
                          '&:hover': {
                            boxShadow: `0 6px 16px ${alpha(currentColor, 0.4)}`,
                          },
                        }}
                      >
                        Continue
                      </Button>
                    </motion.div>
                  ) : (
                    <motion.div
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <Button
                        variant="contained"
                        onClick={handleAnalyze}
                        disabled={analyzing}
                        endIcon={analyzing ? <CircularProgress size={18} color="inherit" /> : <AnalyticsIcon />}
                        disableElevation
                        sx={{
                          borderRadius: 2,
                          py: 1.2,
                          px: 3,
                          background: `linear-gradient(45deg, ${COLOR_PALETTE.success}, ${adjustColor(COLOR_PALETTE.success, 20)})`,
                          fontWeight: 600,
                          boxShadow: `0 4px 12px ${alpha(COLOR_PALETTE.success, 0.3)}`,
                          '&:hover': {
                            boxShadow: `0 6px 16px ${alpha(COLOR_PALETTE.success, 0.4)}`,
                          },
                        }}
                      >
                        {analyzing ? "Processing..." : "Analyze Startup"}
                      </Button>
                    </motion.div>
                  )}
                </Box>
              </Paper>
            </motion.div>
          </AnimatePresence>
          
          {/* Error message */}
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.4 }}
              >
                <Paper
                  elevation={0}
                  sx={{
                    mt: 2,
                    p: 2.5,
                    borderRadius: 2,
                    bgcolor: alpha(COLOR_PALETTE.error, 0.1),
                    border: `1px solid ${alpha(COLOR_PALETTE.error, 0.3)}`,
                    display: 'flex',
                    alignItems: 'center'
                  }}
                >
                  <Typography sx={{ color: COLOR_PALETTE.error, fontWeight: 500 }}>
                    {error}
                  </Typography>
                </Paper>
              </motion.div>
            )}
          </AnimatePresence>
        </Box>
      </Container>
      
      {/* Bottom navigation */}
      <Box sx={{ 
        position: 'fixed', 
        bottom: 0, 
        left: 0, 
        right: 0, 
        py: 2, 
        px: { xs: 2, md: 4 }, 
        bgcolor: 'rgba(255, 255, 255, 0.9)',
        backdropFilter: 'blur(8px)',
        borderTop: '1px solid rgba(0, 0, 0, 0.05)',
        zIndex: 10,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        {/* Left side */}
        <Box>
          {step > 0 ? (
            <Button
              variant="outlined"
              onClick={handlePrev}
              startIcon={<ArrowBackIcon />}
              sx={{
                borderRadius: 2,
                textTransform: 'none',
                borderColor: alpha(COLOR_PALETTE.textLight, 0.3),
                color: COLOR_PALETTE.textLight,
                px: 2,
                py: 1,
                '&:hover': {
                  borderColor: COLOR_PALETTE.textLight,
                  bgcolor: alpha(COLOR_PALETTE.textLight, 0.04)
                }
              }}
            >
              Previous
            </Button>
          ) : (
            <Button
              variant="outlined"
              onClick={() => setForm(populateWithSampleData())}
              startIcon={<RefreshIcon />}
              sx={{
                borderRadius: 2,
                textTransform: 'none',
                borderColor: alpha(COLOR_PALETTE.primary, 0.3),
                color: COLOR_PALETTE.primary,
                px: 2,
                py: 1,
                '&:hover': {
                  borderColor: COLOR_PALETTE.primary,
                  bgcolor: alpha(COLOR_PALETTE.primary, 0.04)
                }
              }}
            >
              Load Sample Data
            </Button>
          )}
        </Box>
        
        {/* Center - Progress */}
        <Box sx={{ display: { xs: 'none', md: 'block' }, width: '50%' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Typography 
              sx={{ 
                mr: 2, 
                color: COLOR_PALETTE.textLight,
                fontSize: 14,
                fontWeight: 500
              }}
            >
              {step + 1} of {ALL_STEPS.length}
            </Typography>
            <Box sx={{ flexGrow: 1, maxWidth: 400 }}>
              <LinearProgress 
                variant="determinate" 
                value={progress} 
                sx={{ 
                  height: 8, 
                  borderRadius: 4,
                  bgcolor: alpha(COLOR_PALETTE.primary, 0.1),
                  '& .MuiLinearProgress-bar': {
                    borderRadius: 4,
                    bgcolor: COLOR_PALETTE.primary
                  }
                }}
              />
            </Box>
          </Box>
        </Box>
        
        {/* Right side */}
        <Box>
          {step < ALL_STEPS.length - 1 ? (
            <Button
              variant="contained"
              onClick={handleNext}
              endIcon={<ArrowForwardIcon />}
              sx={{
                borderRadius: 2,
                textTransform: 'none',
                px: 2.5,
                py: 1,
                boxShadow: `0 4px 14px ${alpha(COLOR_PALETTE.primary, 0.2)}`,
                background: `linear-gradient(135deg, ${COLOR_PALETTE.primary} 0%, ${COLOR_PALETTE.accent1} 100%)`,
                '&:hover': {
                  boxShadow: `0 6px 20px ${alpha(COLOR_PALETTE.primary, 0.3)}`,
                  background: `linear-gradient(135deg, ${COLOR_PALETTE.primary} 0%, ${COLOR_PALETTE.accent1} 80%)`,
                }
              }}
            >
              Next
            </Button>
          ) : (
            <Button
              variant="contained"
              onClick={handleAnalyze}
              disabled={analyzing}
              startIcon={analyzing ? <CircularProgress size={20} color="inherit" /> : <AnalyticsIcon />}
              sx={{
                borderRadius: 2,
                textTransform: 'none',
                px: 2.5,
                py: 1,
                boxShadow: `0 4px 14px ${alpha(COLOR_PALETTE.success, 0.2)}`,
                background: `linear-gradient(135deg, ${COLOR_PALETTE.success} 0%, ${COLOR_PALETTE.accent2} 100%)`,
                '&:hover': {
                  boxShadow: `0 6px 20px ${alpha(COLOR_PALETTE.success, 0.3)}`,
                  background: `linear-gradient(135deg, ${COLOR_PALETTE.success} 0%, ${COLOR_PALETTE.accent2} 80%)`,
                }
              }}
            >
              {analyzing ? "Processing..." : "Analyze"}
            </Button>
          )}
        </Box>
      </Box>
      
      {/* Test controls for development/testing */}
      {process.env.NODE_ENV !== 'production' && renderTestControls()}
    </Box>
  );
};

// Helper function to adjust color brightness
function adjustColor(color: string, amount: number): string {
  return color;
}

export default WizardPage;