#!/usr/bin/env python
"""
Example script demonstrating how to use the hierarchical model architecture in FlashCAMP.
This shows how to make predictions with the model and interpret the results.
"""
import os
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parents[1]))

from flashcamp.backend.app.engines.ml import predict_success_probability
from flashcamp.features.build_features import build_feature_vector

def print_section(title):
    """Print a section header"""
    print("\n" + "=" * 50)
    print(f" {title} ".center(50, "-"))
    print("=" * 50)

def main():
    """Main function to demonstrate hierarchical model usage"""
    print_section("FlashCAMP Hierarchical Model Example")
    
    # Sample startup data for prediction
    startup_data = {
        # Capital metrics
        "cash_on_hand_usd": 3500000,
        "runway_months": 18,
        "burn_multiple": 1.2,
        "ltv_cac_ratio": 3.8,
        "gross_margin_percent": 72,
        "customer_concentration_percent": 15,
        "post_money_valuation_usd": 20000000,
        
        # Advantage metrics
        "patent_count": 3,
        "network_effects_present": True,
        "has_data_moat": True,
        "regulatory_advantage_present": False,
        "tech_differentiation_score": 4.5,
        "switching_cost_score": 3,
        "brand_strength_score": 3,
        "product_retention_30d": 0.88,
        "product_retention_90d": 0.72,
        "nps_score": 55,
        
        # Market metrics
        "tam_size_usd": 7500000000,
        "sam_size_usd": 750000000,
        "claimed_cagr_pct": 28,
        "market_growth_rate_percent": 22,
        "competition_intensity": 2,
        "top3_competitor_share_pct": 50,
        "industry_regulation_level": "low",
        
        # People metrics
        "founders_count": 3,
        "team_size_total": 22,
        "founder_domain_experience_years": 12,
        "prior_successful_exits_count": 1,
        "board_advisor_experience_score": 4,
        "team_diversity_percent": 45,
        "gender_diversity_index": 0.5,
        "geography_diversity_index": 0.4,
        "key_person_dependency": False,
        
        # Additional metrics
        "sector": "FinTech",
        "product_stage": "GA",
        "investor_tier_primary": "Tier1"
    }
    
    print("Analyzing startup data...")
    
    # Make a prediction using the hierarchical model
    result = predict_success_probability(startup_data)
    
    # Extract results
    success_probability = result["success_probability"]
    pillar_scores = result["pillar_scores"]
    
    # Print results
    print_section("Prediction Results")
    print(f"Overall Success Probability: {success_probability:.2%}")
    print("\nPillar Scores:")
    for pillar, score in pillar_scores.items():
        print(f"  {pillar.title()}: {score:.2%}")
    
    # Show pillar contribution analysis
    print_section("Pillar Contribution Analysis")
    
    # Calculate the average pillar score
    avg_score = sum(pillar_scores.values()) / len(pillar_scores)
    print(f"Average Pillar Score: {avg_score:.2%}")
    
    # Identify strengths and weaknesses
    print("\nStrengths and Weaknesses:")
    for pillar, score in sorted(pillar_scores.items(), key=lambda x: x[1], reverse=True):
        if score > avg_score:
            print(f"  ✅ {pillar.title()}: {score:.2%} (+{score - avg_score:.2%} above average)")
        else:
            print(f"  ❌ {pillar.title()}: {score:.2%} ({score - avg_score:.2%} below average)")
    
    # Recommendations based on pillar scores
    print_section("Recommendations")
    weakest_pillar = min(pillar_scores.items(), key=lambda x: x[1])
    print(f"Focus area: {weakest_pillar[0].title()} ({weakest_pillar[1]:.2%})")
    
    if weakest_pillar[0] == "capital":
        print("Capital Recommendations:")
        print("  - Consider raising additional funding to extend runway")
        print("  - Improve unit economics to increase LTV/CAC ratio")
        print("  - Reduce burn rate through operational efficiency")
    elif weakest_pillar[0] == "advantage":
        print("Advantage Recommendations:")
        print("  - Strengthen product differentiation through unique features")
        print("  - Improve customer retention strategies")
        print("  - Consider IP protection strategies like patents")
    elif weakest_pillar[0] == "market":
        print("Market Recommendations:")
        print("  - Expand target market or consider adjacent markets")
        print("  - Develop competitive response strategies")
        print("  - Focus on high-growth market segments")
    elif weakest_pillar[0] == "people":
        print("People Recommendations:")
        print("  - Add domain experts to the team")
        print("  - Improve team diversity")
        print("  - Add experienced advisors")
    
    # Plot pillar scores
    try:
        plt.figure(figsize=(10, 6))
        colors = ['#5DA5DA', '#FAA43A', '#60BD68', '#F17CB0']
        plt.bar(pillar_scores.keys(), pillar_scores.values(), color=colors)
        plt.axhline(y=avg_score, color='red', linestyle='--', label='Average Score')
        plt.ylim(0, 1)
        plt.title('Pillar Scores')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Save the plot
        plot_path = Path(__file__).parent / "pillar_scores_plot.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"\nPillar scores plot saved to: {plot_path}")
    except Exception as e:
        print(f"Error creating plot: {e}")
    
    print_section("Conclusion")
    if success_probability > 0.75:
        print("High potential startup with strong fundamentals across pillars.")
    elif success_probability > 0.5:
        print("Good potential with some areas for improvement.")
    elif success_probability > 0.25:
        print("Moderate potential with significant improvement needed.")
    else:
        print("Low success probability, major improvements required across multiple pillars.")
    
    print("\nDetailed assessment report completed successfully!")

if __name__ == "__main__":
    main() 