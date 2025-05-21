/**
 * Feature flags configuration
 * 
 * This file controls which features are enabled in the FlashCAMP application.
 * Use these flags for progressive rollout of new features.
 */

export interface FeatureFlags {
  /** Enable hierarchical model prediction interface */
  enableHierarchicalModel: boolean;
  
  /** Show advanced model information panel */
  showAdvancedModelInfo: boolean;
  
  /** Enable PDF export with hierarchical model results */
  enableHierarchicalPdfExport: boolean;

  /** Show first-time user onboarding for new features */
  showFeatureOnboarding: boolean;
}

/**
 * Default feature flags for production environment
 */
const productionFlags: FeatureFlags = {
  enableHierarchicalModel: true,      // Hierarchical model is enabled
  showAdvancedModelInfo: true,        // Show advanced model information
  enableHierarchicalPdfExport: false, // PDF export for hierarchical model disabled until ready
  showFeatureOnboarding: false,       // Onboarding disabled until implemented
};

/**
 * Default feature flags for development/staging environment
 */
const developmentFlags: FeatureFlags = {
  enableHierarchicalModel: true,     // Always enabled in development
  showAdvancedModelInfo: true,       // Always enabled in development
  enableHierarchicalPdfExport: true, // Always enabled in development for testing
  showFeatureOnboarding: true,       // Always enabled in development for testing
};

/**
 * Get feature flags based on environment
 */
export function getFeatureFlags(): FeatureFlags {
  // Check for production mode based on available environment variables
  // In a Vite app, this could be different based on the setup
  const isProduction = process.env.NODE_ENV === 'production';
  
  // In production, use production flags
  // In development, use development flags
  return isProduction ? productionFlags : developmentFlags;
}

// Export singleton instance of feature flags
export const featureFlags = getFeatureFlags();

// Export individual flags for convenience
export const {
  enableHierarchicalModel,
  showAdvancedModelInfo,
  enableHierarchicalPdfExport,
  showFeatureOnboarding
} = featureFlags; 