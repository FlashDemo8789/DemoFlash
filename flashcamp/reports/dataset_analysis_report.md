# Dataset Analysis Report

## Basic Information

- Rows: 5000
- Columns: 69
- Missing Values: 0
- Memory Usage: 6.72 MB

## Success Label Information

- Success Rate: None
- Distribution:
  - fail: 3660
  - pass: 1340

## Column Analysis

| Column Name | Type | Missing (%) | Unique Values | Notes |
|------------|------|-------------|---------------|-------|
| startup_id | object | 0.0% | 5000 |  |
| startup_name | object | 0.0% | 5000 |  |
| website | object | 0.0% | 5000 |  |
| sector | object | 0.0% | 12 | Samples: ['ClimateTech', 'BioTech', 'AI', 'Robotics', 'Gaming', 'SaaS', 'EdTech', 'FinTech', 'HealthTech', 'IoT'] |
| subsector | object | 0.0% | 10 | Samples: ['AgriTech', 'LegalTech', 'Payments', 'Cloud', 'DevTools', 'Learning', 'Medtech', 'CleanEnergy', 'DataOps', 'E‑commerce'] |
| market_geography | object | 0.0% | 7 | Samples: ['Africa', 'Global', 'LATAM', 'MENA', 'North America', 'Europe', 'APAC'] |
| country | object | 0.0% | 15 | Samples: ['JPN', 'KOR', 'ITA', 'SGP', 'NLD', 'FRA', 'ZAF', 'CAN', 'DEU', 'BRA'] |
| founding_year | int64 | 0.0% | 7 | Range: 2018.0 - 2024.0 |
| funding_stage | object | 0.0% | 1 | Samples: ['Seed'] |
| total_capital_raised_usd | float64 | 0.0% | 5000 | Range: 145438.8773944445 - 60248770.28182033 |
| funding_rounds_count | int64 | 0.0% | 3 | Range: 1.0 - 3.0 |
| round_valuation_usd | float64 | 0.0% | 5000 | Range: 4007497.271135183 - 26999749.420210376 |
| post_money_valuation_usd | float64 | 0.0% | 5000 | Range: 4728953.9052158175 - 29699724.36223141 |
| committed_funding_usd | float64 | 0.0% | 5000 | Range: 147.9098899776874 - 11343210.802367318 |
| cash_on_hand_usd | float64 | 0.0% | 5000 | Range: 200335.0839951743 - 3698665.1215449967 |
| monthly_burn_usd | float64 | 0.0% | 5000 | Range: 50014.05361349903 - 369986.25326348713 |
| burn_stddev_usd | float64 | 0.0% | 5000 | Range: 2852.697602842254 - 127587.64574200062 |
| annual_revenue_run_rate | float64 | 0.0% | 5000 | Range: 1428.521484128209 - 3899715.300402895 |
| revenue_growth_rate_percent | float64 | 0.0% | 5000 | Range: -29.99305239611396 - 379.92890665888586 |
| ltv_cac_ratio | float64 | 0.0% | 5000 | Range: 0.4012882821461364 - 5.999405021747326 |
| cac_usd | float64 | 0.0% | 5000 | Range: 401.7061678850448 - 14994.105837083096 |
| gross_margin_percent | float64 | 0.0% | 5000 | Range: 25.007983331810586 - 93.97371327793866 |
| customer_churn_rate_percent | float64 | 0.0% | 4793 | Range: 1.0 - 24.49991285638984 |
| has_debt | int64 | 0.0% | 2 | Range: 0.0 - 1.0 |
| investor_tier_primary | object | 0.0% | 4 | Samples: ['Tier2', 'Unknown', 'Tier1', 'Angel'] |
| customer_concentration_percent | float64 | 0.0% | 5000 | Range: 0.0041009662108821 - 71.98736529036181 |
| patent_count | int64 | 0.0% | 9 | Range: 0.0 - 8.0 |
| network_effects_present | int64 | 0.0% | 2 | Range: 0.0 - 1.0 |
| has_data_moat | int64 | 0.0% | 2 | Range: 0.0 - 1.0 |
| regulatory_advantage_present | int64 | 0.0% | 2 | Range: 0.0 - 1.0 |
| scalability_score | float64 | 0.0% | 5000 | Range: 1.000049384638534 - 4.999747015614904 |
| tech_differentiation_score | float64 | 0.0% | 5000 | Range: 1.0003041903699392 - 4.999699552280849 |
| switching_cost_score | float64 | 0.0% | 5000 | Range: 1.001794666682046 - 4.999645571528179 |
| brand_strength_score | float64 | 0.0% | 5000 | Range: 1.0032361623537764 - 4.9987029934785685 |
| product_stage | object | 0.0% | 3 | Samples: ['Concept', 'Beta', 'GA'] |
| customer_count | int64 | 0.0% | 461 | Range: 1.0 - 3130.0 |
| net_dollar_retention_percent | float64 | 0.0% | 4926 | Range: 50.0 - 228.5473841715896 |
| mau | float64 | 0.0% | 5000 | Range: 2.6266341708644942 - 7472.223144570035 |
| product_retention_30d | float64 | 0.0% | 5000 | Range: 0.0500292780737086 - 0.8996903279168458 |
| nps_score | float64 | 0.0% | 5000 | Range: -59.96935707234263 - 94.96961459916535 |
| tam_size_usd | float64 | 0.0% | 5000 | Range: 106229642.79150496 - 15999054651.869593 |
| sam_size_usd | float64 | 0.0% | 5000 | Range: 11933660.527891016 - 7857825291.735383 |
| tam_justification | object | 0.0% | 1 | Samples: ['Hybrid estimate'] |
| claimed_cagr_pct | float64 | 0.0% | 5000 | Range: 4.013792163183013 - 49.99453990406235 |
| market_growth_rate_percent | float64 | 0.0% | 5000 | Range: -11.985687528961432 - 67.99966371450155 |
| user_growth_rate_percent | float64 | 0.0% | 5000 | Range: -34.96007679629494 - 379.9078204918566 |
| product_retention_90d | float64 | 0.0% | 5000 | Range: 0.020007663451116 - 0.7998308438854334 |
| top3_competitor_share_pct | float64 | 0.0% | 5000 | Range: 0.0500801148038988 - 0.9499952929506248 |
| competition_intensity | float64 | 0.0% | 5000 | Range: 1.0003722690568586 - 4.999353570431392 |
| competitors_named_count | int64 | 0.0% | 36 | Range: 0.0 - 35.0 |
| industry_regulation_level | int64 | 0.0% | 4 | Range: 0.0 - 3.0 |
| regulation_flags_json | object | 0.0% | 8 | Samples: ['{"gdpr": 1, "ccpa": 1, "sox": 1}', '{"gdpr": 0, "ccpa": 0, "sox": 0}', '{"gdpr": 0, "ccpa": 1, "sox": 1}', '{"gdpr": 0, "ccpa": 1, "sox": 0}', '{"gdpr": 1, "ccpa": 0, "sox": 1}', '{"gdpr": 1, "ccpa": 0, "sox": 0}', '{"gdpr": 1, "ccpa": 1, "sox": 0}', '{"gdpr": 0, "ccpa": 0, "sox": 1}'] |
| countries_served_json | object | 0.0% | 3420 |  |
| conversion_rate_percent | float64 | 0.0% | 5000 | Range: 1.004813441709859 - 24.995860066220395 |
| founders_count | int64 | 0.0% | 5 | Range: 1.0 - 5.0 |
| team_size_total | int64 | 0.0% | 196 | Range: 5.0 - 200.0 |
| team_size_full_time | int64 | 0.0% | 192 | Range: 3.0 - 195.0 |
| founder_domain_experience_years | int64 | 0.0% | 19 | Range: 0.0 - 18.0 |
| prior_startup_experience_count | int64 | 0.0% | 4 | Range: 0.0 - 3.0 |
| prior_successful_exits_count | int64 | 0.0% | 4 | Range: 0.0 - 3.0 |
| board_advisor_experience_score | float64 | 0.0% | 5000 | Range: 7.209254224704864e-06 - 4.999729021638465 |
| advisors_count | int64 | 0.0% | 16 | Range: 0.0 - 15.0 |
| team_diversity_percent | float64 | 0.0% | 5000 | Range: 0.0125693478262856 - 99.9821567519034 |
| gender_diversity_index | float64 | 0.0% | 5000 | Range: 0.0009095410877201 - 0.999806090109381 |
| linkedin_connections_total | int64 | 0.0% | 4650 | Range: 3.0 - 34997.0 |
| twitter_followers_total | int64 | 0.0% | 4953 | Range: 11.0 - 219976.0 |
| key_person_dependency | int64 | 0.0% | 2 | Range: 0.0 - 1.0 |
| geography_diversity_index | float64 | 0.0% | 5000 | Range: 4.5599966569054295e-05 - 0.9997167878679808 |
| success_label | object | 0.0% | 2 | Samples: ['fail', 'pass'] |
