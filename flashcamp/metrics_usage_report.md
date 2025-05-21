# Metrics Usage Report

Total metrics: 69
Metrics used in components: 17
Metrics not used in any component: 52

## Usage by Component

### frontend/src/components/PillarStep.tsx
Uses 0 metrics:


### frontend/src/components/PDFDownloadButton.tsx
Uses 3 metrics:

**Info**: sector, startup_id, startup_name

### backend/app/engines/ml.py
Uses 15 metrics:

**Advantage**: network_effects_present, switching_cost_score, tech_differentiation_score
**Capital**: cash_on_hand_usd, monthly_burn_usd
**Info**: funding_stage, startup_id
**Market**: competition_intensity, market_growth_rate_percent, tam_size_usd
**People**: founder_domain_experience_years, prior_startup_experience_count, prior_successful_exits_count, team_diversity_percent, team_size_full_time

## Unused Metrics

**Advantage**: brand_strength_score, has_data_moat, patent_count, product_stage, regulatory_advantage_present, scalability_score
**Capital**: annual_revenue_run_rate, burn_stddev_usd, cac_usd, committed_funding_usd, customer_churn_rate_percent, customer_concentration_percent, funding_rounds_count, gross_margin_percent, has_debt, investor_tier_primary, ltv_cac_ratio, net_dollar_retention_percent, post_money_valuation_usd, revenue_growth_rate_percent, round_valuation_usd, total_capital_raised_usd
**Info**: country, founding_year, market_geography, subsector, website
**Market**: claimed_cagr_pct, competitors_named_count, conversion_rate_percent, countries_served_json, customer_count, industry_regulation_level, mau, nps_score, product_retention_30d, product_retention_90d, regulation_flags_json, sam_size_usd, tam_justification, top3_competitor_share_pct, user_growth_rate_percent
**People**: advisors_count, board_advisor_experience_score, founders_count, gender_diversity_index, geography_diversity_index, key_person_dependency, linkedin_connections_total, team_size_total, twitter_followers_total
**PredictiveLabel**: success_label