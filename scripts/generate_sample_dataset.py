#!/usr/bin/env python
"""
Generate a sample dataset for training hierarchical models.
This creates a synthetic dataset with the required features and labels
for training the pillar models and meta-model.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import random

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parents[1]))

def generate_synthetic_startup(success_prob=None):
    """Generate synthetic data for a single startup"""
    # If success_prob is provided, bias the metrics to reflect this probability
    # Otherwise, assign a random success probability
    if success_prob is None:
        success_prob = random.random()
    
    # Base parameter - controls feature quality based on success probability
    quality = success_prob
    
    # Add some noise to make the data more realistic
    def add_noise(base_value, noise_level=0.2):
        return base_value * (1 + noise_level * (random.random() - 0.5))
    
    # Capital metrics - financial health
    cash_on_hand = add_noise(quality * 5000000, 0.5)
    runway_months = add_noise(quality * 24, 0.3)
    burn_multiple = add_noise((1 - quality) * 3 + 0.5, 0.3)  # Lower is better
    ltv_cac_ratio = add_noise(quality * 5, 0.4)
    gross_margin = add_noise(quality * 80, 0.3)
    customer_concentration = add_noise((1 - quality) * 60, 0.5)  # Lower is better
    post_money_valuation = add_noise(quality * 30000000, 0.6)
    
    # Advantage metrics - product differentiation
    patent_count = int(add_noise(quality * 5, 0.7))
    network_effects = random.random() < (quality * 0.8 + 0.1)
    data_moat = random.random() < (quality * 0.7 + 0.2)
    regulatory_advantage = random.random() < (quality * 0.5 + 0.1)
    tech_diff_score = add_noise(quality * 5, 0.3)
    switching_cost = add_noise(quality * 5, 0.4)
    brand_strength = add_noise(quality * 5, 0.5)
    retention_30d = add_noise(quality * 0.9 + 0.05, 0.2)
    retention_90d = add_noise(quality * 0.7 + 0.05, 0.3)
    nps_score = add_noise(quality * 80 - 40, 0.4)  # Scale from -100 to 100
    
    # Market metrics - market opportunity
    tam_size = add_noise(quality * 10000000000, 0.7)
    sam_size = add_noise(tam_size * 0.1, 0.3)
    cagr_pct = add_noise(quality * 40, 0.4)
    market_growth = add_noise(quality * 30, 0.5)
    competition = add_noise((1 - quality) * 5, 0.4)  # Lower is better
    top3_competitor_share = add_noise((1 - quality) * 80 + 10, 0.3)
    
    # Set regulation level based on quality
    reg_level_probs = [0.1, 0.2, 0.4, 0.2, 0.1]  # Probabilities for different levels
    if quality < 0.3:
        reg_level_probs = [0.05, 0.1, 0.2, 0.3, 0.35]  # Higher regulation for low quality
    elif quality > 0.7:
        reg_level_probs = [0.3, 0.3, 0.25, 0.1, 0.05]  # Lower regulation for high quality
    
    regulation_levels = ["very_low", "low", "medium", "high", "very_high"]
    regulation_level = np.random.choice(regulation_levels, p=reg_level_probs)
    
    # People metrics - team composition
    founders_count = max(1, int(add_noise(quality * 3 + 1, 0.5)))
    team_size = max(1, int(add_noise(quality * 40 + 5, 0.6)))
    domain_experience = add_noise(quality * 15, 0.4)
    prior_exits = max(0, int(add_noise(quality * 2, 0.7)))
    advisor_score = add_noise(quality * 5, 0.3)
    team_diversity = add_noise(quality * 60, 0.4)
    gender_diversity = add_noise(quality * 0.8 + 0.1, 0.3)
    geo_diversity = add_noise(quality * 0.6 + 0.1, 0.5)
    key_person_dependency = random.random() > (quality * 0.7 + 0.1)  # Less dependency is better
    
    # Sector selection with appropriate distribution
    sectors = ["FinTech", "HealthTech", "EdTech", "SaaS", "AI", "BioTech", "ClimateTech", "Gaming", "Robotics", "IoT", "Other"]
    sector_probs = [0.15, 0.12, 0.10, 0.20, 0.15, 0.08, 0.05, 0.05, 0.04, 0.04, 0.02]
    sector = np.random.choice(sectors, p=sector_probs)
    
    # Product stage based on quality
    stages = ["Concept", "Beta", "GA"]
    stage_probs = [0.2, 0.3, 0.5]  # Default distribution
    if quality < 0.3:
        stage_probs = [0.6, 0.3, 0.1]  # Early stage for low quality
    elif quality > 0.7:
        stage_probs = [0.1, 0.2, 0.7]  # Later stage for high quality
    
    product_stage = np.random.choice(stages, p=stage_probs)
    
    # Investor tier based on quality
    tiers = ["Unknown", "Angel", "Tier2", "Tier1"]
    tier_probs = [0.25, 0.25, 0.25, 0.25]  # Default distribution
    if quality < 0.3:
        tier_probs = [0.5, 0.3, 0.15, 0.05]  # Lower tier for low quality
    elif quality > 0.7:
        tier_probs = [0.1, 0.2, 0.3, 0.4]  # Higher tier for high quality
    
    investor_tier = np.random.choice(tiers, p=tier_probs)
    
    # Create success label - binary classification with probability based on quality
    # Add some randomness to make it more realistic
    success_threshold = 0.5
    success_dice = random.random() * 0.3 + quality * 0.7  # Blend randomness with quality
    success_label = "pass" if success_dice > success_threshold else "fail"
    
    # Create the startup data dictionary
    startup = {
        # Capital metrics
        "cash_on_hand_usd": cash_on_hand,
        "runway_months": runway_months,
        "burn_multiple": burn_multiple,
        "ltv_cac_ratio": ltv_cac_ratio,
        "gross_margin_percent": gross_margin,
        "customer_concentration_percent": customer_concentration,
        "post_money_valuation_usd": post_money_valuation,
        
        # Advantage metrics
        "patent_count": patent_count,
        "network_effects_present": network_effects,
        "has_data_moat": data_moat,
        "regulatory_advantage_present": regulatory_advantage,
        "tech_differentiation_score": tech_diff_score,
        "switching_cost_score": switching_cost,
        "brand_strength_score": brand_strength,
        "product_retention_30d": retention_30d,
        "product_retention_90d": retention_90d,
        "nps_score": nps_score,
        
        # Market metrics
        "tam_size_usd": tam_size,
        "sam_size_usd": sam_size,
        "claimed_cagr_pct": cagr_pct,
        "market_growth_rate_percent": market_growth,
        "competition_intensity": competition,
        "top3_competitor_share_pct": top3_competitor_share,
        "industry_regulation_level": regulation_level,
        
        # People metrics
        "founders_count": founders_count,
        "team_size_total": team_size,
        "founder_domain_experience_years": domain_experience,
        "prior_successful_exits_count": prior_exits,
        "board_advisor_experience_score": advisor_score,
        "team_diversity_percent": team_diversity,
        "gender_diversity_index": gender_diversity,
        "geography_diversity_index": geo_diversity,
        "key_person_dependency": key_person_dependency,
        
        # Additional metrics
        "sector": sector,
        "product_stage": product_stage,
        "investor_tier_primary": investor_tier,
        
        # Success label - ground truth for training
        "success_label": success_label
    }
    
    return startup

def generate_dataset(num_samples=5000, success_rate=0.38):
    """Generate a complete dataset with specified number of samples and success rate"""
    startups = []
    
    # Calculate number of successful startups based on desired success rate
    num_success = int(num_samples * success_rate)
    num_failure = num_samples - num_success
    
    # Generate successful startups (biased towards higher quality)
    for _ in range(num_success):
        # Generate quality biased towards higher values for successful startups
        quality = random.random() * 0.4 + 0.6  # Range 0.6-1.0
        startup = generate_synthetic_startup(quality)
        startup["success_label"] = "pass"
        startups.append(startup)
    
    # Generate failed startups (biased towards lower quality)
    for _ in range(num_failure):
        # Generate quality biased towards lower values for failed startups
        quality = random.random() * 0.5  # Range 0-0.5
        startup = generate_synthetic_startup(quality)
        startup["success_label"] = "fail"
        startups.append(startup)
    
    # Convert to DataFrame and shuffle
    df = pd.DataFrame(startups)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def main():
    """Main function to generate and save the dataset"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic startup dataset")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples to generate")
    parser.add_argument("--success_rate", type=float, default=0.38, help="Success rate (proportion of 'pass' labels)")
    parser.add_argument("--output", type=str, default="flashcamp/data/gold/seed_dataset_synthetic.csv", 
                        help="Output file path")
    args = parser.parse_args()
    
    print(f"Generating synthetic dataset with {args.samples} samples...")
    df = generate_dataset(args.samples, args.success_rate)
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Actual success rate: {df['success_label'].value_counts(normalize=True)['pass']:.2%}")

if __name__ == "__main__":
    main() 