"""
Transforms a validated StartupInput dict into a flat NumPy vector.
The list FEATURE_COLUMNS defines stable name → index order so the
same vector feeds both training and FastAPI inference.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any
from .encode import (
    SWITCHING_COST_IDX, 
    REG_LEVEL_IDX, 
    DEGREE_IDX, 
    INVESTOR_TIER_IDX,
    PRODUCT_STAGE_IDX,
    safe_div, 
    one_hot,
    parse_json_field
)

# ---------------- Fixed domains for one-hot -----------------
SECTORS = (
    "FinTech", "HealthTech", "EdTech", "SaaS", "AI", 
    "BioTech", "ClimateTech", "Gaming", "Robotics", "IoT", "Other"
)
TAM_METHOD  = ("Hybrid estimate",)
PRODUCT_STAGES = ("Concept", "Beta", "GA")
INVESTOR_TIERS = ("Unknown", "Angel", "Tier2", "Tier1")

# ---------------- Column ordering ----------------------------
FEATURE_COLUMNS: list[str] = [
    # Capital ratios
    "log_cash_on_hand", "ltv_cac_ratio", "burn_multiple", "runway_est",
    "gross_margin", "customer_concentration", "post_money_valuation",
    
    # Advantage
    "patent_count_norm", "network_effect", "has_data_moat", "reg_advantage",
    "tech_diff_score", "switch_cost_score", "brand_strength",
    "retention_30d", "retention_90d", "nps_score_norm",
    
    # Market
    "tam_ratio", "sam_ratio", "cagr_pct", "market_growth_pct",
    "competition_intensity", "competition_hhi", "reg_level_idx",
    
    # People
    "founders_count_norm", "team_size_norm", "domain_exp_avg",
    "prior_exits", "board_advisor_score", "team_diversity",
    "gender_div_idx", "geo_div_idx", "key_person_dependency",
    
    # Social
    "log_linkedin", "log_twitter", "advisors_count_norm",
    
    # Performance
    "conversion_rate", "user_growth_rate", "customer_count_norm",
    "scalability_score", "net_dollar_retention",
    
    # One-hots
    *[f"sector_{x}" for x in SECTORS],
    *[f"tam_method_{x}" for x in TAM_METHOD],
    *[f"product_stage_{x}" for x in PRODUCT_STAGES],
    *[f"investor_tier_{x}" for x in INVESTOR_TIERS],
]

def build_feature_vector(d: Dict[str, Any]) -> np.ndarray:
    """
    Transform a dictionary of startup metrics into a feature vector.
    The input dictionary should match the structure defined in contracts/metrics.json.
    """
    # Handle boolean values that might be represented as 0/1 in the dataset
    for key in ["network_effects_present", "has_data_moat", "regulatory_advantage_present", "key_person_dependency", "has_debt"]:
        if key in d and isinstance(d[key], (int, float)):
            d[key] = bool(d[key])
    
    # ---- Capital -------------------------------------------
    log_cash = np.log1p(d.get("cash_on_hand_usd", 0))
    ltv_cac_ratio = d.get("ltv_cac_ratio", 0)
    burn_multiple = safe_div(d.get("monthly_burn_usd", 0) * 12, d.get("annual_revenue_run_rate", 1e-6))
    runway_est = safe_div(d.get("cash_on_hand_usd", 0), d.get("monthly_burn_usd", 1e-6))
    gross_margin = d.get("gross_margin_percent", 0) / 100
    customer_concentration = d.get("customer_concentration_percent", 0) / 100
    post_money_valuation = np.log1p(d.get("post_money_valuation_usd", 0)) / 20  # Normalize large values
    
    # ---- Advantage -----------------------------------------
    patent_count_norm = d.get("patent_count", 0) / 10  # Normalize patent count
    network_effect = 1.0 if d.get("network_effects_present", False) else 0.0
    has_data_moat = 1.0 if d.get("has_data_moat", False) else 0.0
    reg_advantage = 1.0 if d.get("regulatory_advantage_present", False) else 0.0
    tech_diff_score = d.get("tech_differentiation_score", 0) / 5  # Normalize 0-5 scale
    switch_cost_score = d.get("switching_cost_score", 0) / 5  # Normalize 0-5 scale
    brand_strength = d.get("brand_strength_score", 0) / 5  # Normalize 0-5 scale
    retention_30d = d.get("product_retention_30d", 0)  # Already 0-1
    retention_90d = d.get("product_retention_90d", 0)  # Already 0-1
    nps_score_norm = (d.get("nps_score", 0) + 100) / 200  # Normalize -100 to 100 scale to 0-1
    
    # ---- Market --------------------------------------------
    baseline_tam = 1e10  # Higher baseline for normalization
    tam_ratio = safe_div(d.get("tam_size_usd", 0), baseline_tam)
    sam_ratio = safe_div(d.get("sam_size_usd", 0), baseline_tam / 10)  # SAM should be smaller than TAM
    cagr_pct = d.get("claimed_cagr_pct", 0) / 100
    market_growth_pct = d.get("market_growth_rate_percent", 0) / 100
    competition_intensity = d.get("competition_intensity", 0) / 5  # Normalize 0-5 scale
    competition_hhi = (d.get("top3_competitor_share_pct", 0) / 100) ** 2 * 3  # HHI index approximation
    
    # Process regulation flags JSON to extract count of active regulations
    reg_flags = parse_json_field(d.get("regulation_flags_json"), {})
    reg_count = sum(1 for v in reg_flags.values() if v == 1)
    
    # Get regulation level index
    reg_level = d.get("industry_regulation_level", 0)
    reg_level_idx = REG_LEVEL_IDX.get(reg_level, 0.0)
    
    # ---- People --------------------------------------------
    founders_count_norm = d.get("founders_count", 0) / 5  # Normalize by max expected value
    team_size_norm = d.get("team_size_total", 0) / 200  # Normalize by max expected value
    domain_exp_avg = d.get("founder_domain_experience_years", 0) / 20  # Normalize years
    prior_exits = d.get("prior_successful_exits_count", 0) / 3  # Normalize by max expected value
    board_advisor_score = d.get("board_advisor_experience_score", 0)  # Already 0-5 scale
    team_diversity = d.get("team_diversity_percent", 0) / 100  # Normalize to 0-1
    gender_div_idx = d.get("gender_diversity_index", 0)  # Already 0-1
    geo_div_idx = d.get("geography_diversity_index", 0)  # Already 0-1
    key_person_dependency = 1.0 if d.get("key_person_dependency", False) else 0.0
    
    # ---- Social --------------------------------------------
    log_linkedin = np.log1p(d.get("linkedin_connections_total", 0)) / 15  # Normalize log value
    log_twitter = np.log1p(d.get("twitter_followers_total", 0)) / 15  # Normalize log value
    advisors_count_norm = d.get("advisors_count", 0) / 15  # Normalize by max expected value
    
    # ---- Performance --------------------------------------
    conversion_rate = d.get("conversion_rate_percent", 0) / 100  # Normalize to 0-1
    user_growth_rate = d.get("user_growth_rate_percent", 0) / 400  # Normalize to 0-1 (max ~380%)
    customer_count_norm = np.log1p(d.get("customer_count", 0)) / 10  # Log normalize
    scalability_score = d.get("scalability_score", 0) / 5  # Normalize 0-5 scale
    net_dollar_retention = d.get("net_dollar_retention_percent", 100) / 200  # Normalize to 0-1
    
    # ---- One-hot encodings ---------------------------------
    # Get sector (with fallback to "Other")
    sector = d.get("sector", "Other") 
    if sector not in SECTORS:
        sector = "Other"
    
    # Get product stage (with default to first value)
    product_stage = d.get("product_stage", PRODUCT_STAGES[0])
    
    # Get investor tier (with default)
    investor_tier = d.get("investor_tier_primary", "Unknown")
    
    # Get TAM justification (with default)
    tam_method = d.get("tam_justification", TAM_METHOD[0])
    
    # ---- Assemble the vector ------------------------------
    vec = [
        # Capital
        log_cash, ltv_cac_ratio, burn_multiple, runway_est,
        gross_margin, customer_concentration, post_money_valuation,
        
        # Advantage
        patent_count_norm, network_effect, has_data_moat, reg_advantage,
        tech_diff_score, switch_cost_score, brand_strength,
        retention_30d, retention_90d, nps_score_norm,
        
        # Market
        tam_ratio, sam_ratio, cagr_pct, market_growth_pct,
        competition_intensity, competition_hhi, reg_level_idx,
        
        # People
        founders_count_norm, team_size_norm, domain_exp_avg,
        prior_exits, board_advisor_score, team_diversity,
        gender_div_idx, geo_div_idx, key_person_dependency,
        
        # Social
        log_linkedin, log_twitter, advisors_count_norm,
        
        # Performance
        conversion_rate, user_growth_rate, customer_count_norm,
        scalability_score, net_dollar_retention,
        
        # One-hot encodings
        *one_hot(sector, SECTORS),
        *one_hot(tam_method, TAM_METHOD),
        *one_hot(product_stage, PRODUCT_STAGES),
        *one_hot(investor_tier, INVESTOR_TIERS),
    ]
    
    return np.asarray(vec).reshape(1, -1) 