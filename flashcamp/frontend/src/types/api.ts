/**
 * API Types
 * 
 * This file defines TypeScript interfaces for API responses from the FlashCAMP backend.
 */

/**
 * An alert represents a warning or insight about the startup analysis
 */
export interface Alert {
  type: string;
  message: string;
  severity: string;
}

/**
 * Pillar scores represent the performance in each of the four CAMP pillars
 */
export interface PillarScores {
  capital: number;
  advantage: number;
  market: number;
  people: number;
  [key: string]: number; // Allow additional pillar scores
}

/**
 * Analysis result from the `/api/analyze` endpoint
 */
export interface AnalysisResult {
  // Primary scores
  success_probability: number;
  capital_score: number;
  advantage_score: number;
  market_score: number;
  people_score: number;
  overall_score: number;
  
  // V2 model detailed pillar scores
  pillar_scores?: PillarScores;
  
  // Alerts and recommendations
  alerts: Alert[];
  insights?: any[];
  recommendations?: any[];
  
  // Startup metadata
  startup_id: string;
  startup_name: string;
  sector?: string;
  founding_year?: number;
  team_size_total?: number;
  funding_stage?: string;
  runway_months?: number;
  product_stage?: string;
  
  // Error handling
  error?: string | null;
  
  // Allow additional properties
  [key: string]: any;
}

/**
 * Runway simulation result from the `/api/runway_sim` endpoint
 */
export interface RunwaySimResult {
  months: number[];
  cash: number[];
  runway_months: number;
}

/**
 * Report generation response from the `/api/generate_report` endpoint
 */
export interface ReportResponse {
  file_url: string;
  success: boolean;
  message?: string;
}

/**
 * Hierarchical model prediction response from the `/api/prediction/predict` endpoint
 */
export interface HierarchicalPredictionResponse {
  pillar_scores: PillarScores;
  final_score: number;
  prediction: 'pass' | 'fail';
  confidence: number;
  threshold: number;
  error?: string | null;
}

/**
 * A recommendation item for improving startup success
 */
export interface RecommendationItem {
  metric: string;
  recommendation: string;
  impact: 'high' | 'medium' | 'low';
}

/**
 * Recommendations response from the `/api/prediction/recommendations` endpoint
 */
export interface RecommendationsResponse {
  capital: RecommendationItem[];
  advantage: RecommendationItem[];
  market: RecommendationItem[];
  people: RecommendationItem[];
}

/**
 * Model performance metrics
 */
export interface ModelMetrics {
  auc?: number;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  calibration_error?: number;
}

/**
 * Model information response from the `/api/prediction/model-info` endpoint
 */
export interface ModelInfoResponse {
  model_version: string;
  dataset_size: number;
  success_rate: number;
  threshold: number;
  pillar_metrics: {
    capital?: ModelMetrics;
    advantage?: ModelMetrics;
    market?: ModelMetrics;
    people?: ModelMetrics;
    [key: string]: ModelMetrics | undefined;
  };
  meta_metrics: ModelMetrics;
} 