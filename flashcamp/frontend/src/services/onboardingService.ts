/**
 * Onboarding Service
 * 
 * Handles user onboarding preferences and storage
 */

// Keys for localStorage
const STORAGE_KEYS = {
  HIERARCHICAL_MODEL_ONBOARDING_SHOWN: 'flash_hierarchical_onboarding_shown',
  HIERARCHICAL_MODEL_ONBOARDING_COMPLETED: 'flash_hierarchical_onboarding_completed',
};

/**
 * Check if hierarchical model onboarding should be shown
 * @returns boolean indicating if onboarding should be shown
 */
export function shouldShowHierarchicalOnboarding(): boolean {
  // Check if user has seen or completed the onboarding
  const onboardingShown = localStorage.getItem(STORAGE_KEYS.HIERARCHICAL_MODEL_ONBOARDING_SHOWN);
  const onboardingCompleted = localStorage.getItem(STORAGE_KEYS.HIERARCHICAL_MODEL_ONBOARDING_COMPLETED);
  
  // Don't show if user has completed onboarding
  if (onboardingCompleted === 'true') {
    return false;
  }
  
  // Don't show if user has seen onboarding in this session
  if (onboardingShown === 'true') {
    return false;
  }
  
  // Show onboarding by default for new users
  return true;
}

/**
 * Mark hierarchical model onboarding as shown for this session
 */
export function markHierarchicalOnboardingAsShown(): void {
  localStorage.setItem(STORAGE_KEYS.HIERARCHICAL_MODEL_ONBOARDING_SHOWN, 'true');
}

/**
 * Mark hierarchical model onboarding as completed (user won't see it again)
 */
export function markHierarchicalOnboardingAsCompleted(): void {
  localStorage.setItem(STORAGE_KEYS.HIERARCHICAL_MODEL_ONBOARDING_COMPLETED, 'true');
}

/**
 * Reset onboarding preferences (for testing)
 */
export function resetOnboardingPreferences(): void {
  localStorage.removeItem(STORAGE_KEYS.HIERARCHICAL_MODEL_ONBOARDING_SHOWN);
  localStorage.removeItem(STORAGE_KEYS.HIERARCHICAL_MODEL_ONBOARDING_COMPLETED);
} 