/**
 * Hierarchical Model API Service
 * 
 * This service handles API calls to the hierarchical model endpoints.
 */
import {
  HierarchicalPredictionResponse,
  RecommendationsResponse,
  ModelInfoResponse
} from '../types/api';

// API base URL - adjust for production environment as needed
// For Vite apps, we use import.meta.env.VITE_* for environment variables
const API_BASE_URL = 'http://localhost:8000';

/**
 * Get prediction for a startup using the hierarchical model
 * 
 * @param startupData - The startup metrics data
 * @returns Promise with the prediction response
 */
export async function getPrediction(startupData: Record<string, any>): Promise<HierarchicalPredictionResponse> {
  try {
    console.log('Calling prediction endpoint with data:', startupData);
    const response = await fetch(`${API_BASE_URL}/api/prediction/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(startupData),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Error fetching prediction: ${response.status} ${response.statusText}\n${errorText}`);
    }

    const data = await response.json();
    console.log('Prediction response:', data);
    return data;
  } catch (error) {
    console.error('Error getting prediction:', error);
    throw error;
  }
}

/**
 * Get recommendations for improving startup success probability
 * 
 * @param startupData - The startup metrics data
 * @returns Promise with the recommendations response
 */
export async function getRecommendations(startupData: Record<string, any>): Promise<RecommendationsResponse> {
  try {
    console.log('Calling recommendations endpoint with data:', startupData);
    const response = await fetch(`${API_BASE_URL}/api/prediction/recommendations`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(startupData),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Error fetching recommendations: ${response.status} ${response.statusText}\n${errorText}`);
    }

    const data = await response.json();
    console.log('Recommendations response:', data);
    return data;
  } catch (error) {
    console.error('Error getting recommendations:', error);
    throw error;
  }
}

/**
 * Get visualization URL for the prediction results
 * This returns the URL that can be used as an image source
 * 
 * @param startupData - The startup metrics data
 * @returns String URL to the visualization image
 */
export function getVisualizationUrl(startupData: Record<string, any>): string {
  // Create a unique timestamp to prevent caching
  const timestamp = new Date().getTime();
  
  // For GET requests with many parameters, we'll use POST instead
  // Just return the URL with a cache-busting parameter
  return `${API_BASE_URL}/api/prediction/visualization?_t=${timestamp}`;
}

/**
 * Get visualization as a blob (for direct loading or saving)
 * 
 * @param startupData - The startup metrics data
 * @returns Promise with the blob data
 */
export async function getVisualizationBlob(startupData: Record<string, any>): Promise<Blob> {
  try {
    console.log('Calling visualization endpoint with data:', startupData);
    const response = await fetch(`${API_BASE_URL}/api/prediction/visualization`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(startupData),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Error fetching visualization: ${response.status} ${response.statusText}\n${errorText}`);
    }

    return await response.blob();
  } catch (error) {
    console.error('Error getting visualization:', error);
    throw error;
  }
}

/**
 * Get model information including performance metrics
 * 
 * @returns Promise with the model info response
 */
export async function getModelInfo(): Promise<ModelInfoResponse> {
  try {
    console.log('Calling model-info endpoint');
    const response = await fetch(`${API_BASE_URL}/api/prediction/model-info`);

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Error fetching model info: ${response.status} ${response.statusText}\n${errorText}`);
    }

    const data = await response.json();
    console.log('Model info response:', data);
    return data;
  } catch (error) {
    console.error('Error getting model info:', error);
    throw error;
  }
}

/**
 * Combined function to get all hierarchical model data in a single call
 * (reduces number of requests when multiple pieces of information are needed)
 * 
 * @param startupData - The startup metrics data
 * @returns Promise with combined prediction and recommendations data
 */
export async function getHierarchicalAnalysis(startupData: Record<string, any>): Promise<{
  prediction: HierarchicalPredictionResponse;
  recommendations: RecommendationsResponse;
}> {
  try {
    console.log('Getting hierarchical analysis for data:', startupData);
    // Fetch both prediction and recommendations in parallel
    const [predictionResponse, recommendationsResponse] = await Promise.all([
      getPrediction(startupData),
      getRecommendations(startupData)
    ]);

    return {
      prediction: predictionResponse,
      recommendations: recommendationsResponse
    };
  } catch (error) {
    console.error('Error getting hierarchical analysis:', error);
    throw error;
  }
} 