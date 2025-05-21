# AUTO-GENERATED from contracts/metrics.json - DO NOT EDIT MANUALLY
from typing import List, Any, Dict, Optional, Union
from pydantic import BaseModel, Field


class MetricsInput(BaseModel):
    """Input model for startup metrics"""
    startup_name: Optional[str] = Field(default='', description="Startup Name")
    startup_id: Optional[str] = Field(default='', description="Startup Id")
    sector: Optional[str] = Field(default='', description="Sector")
    subsector: Optional[str] = Field(default='', description="Subsector")
    market_geography: Optional[List[str]] = Field(default=[], description="Market Geography")
    country: Optional[str] = Field(default='', description="Country")
    website: Optional[str] = Field(default='', description="Website")
    founding_year: Optional[float] = Field(default=2022, description="Founding Year")
    funding_stage: Optional[str] = Field(default='Seed', description="Funding Stage")
    has_data_moat: Optional[bool] = Field(default=False, description="Has Data Moat")
    prior_startup_experience_count: Optional[float] = Field(default=0, description="Prior Startup Experience Count")
    runway_months: Optional[float] = Field(default=12, description="Runway (Months)")
    tam_size_usd: Optional[float] = Field(default=0, description="TAM Size (USD)")
    regulatory_risk_score: Optional[float] = Field(default=5, description="Regulatory Risk Score")
    market_growth_rate_percent: Optional[float] = Field(default=0, description="Market Growth Rate (%)")
    conversion_rate_percent: Optional[float] = Field(default=0, description="Conversion Rate (%)")
    board_advisor_experience_score: Optional[float] = Field(default=0, description="Board Advisor Experience Score")
    tech_differentiation_score: Optional[float] = Field(default=0, description="Tech Differentiation Score")
    regulatory_advantage_present: Optional[bool] = Field(default=False, description="Regulatory Advantage Present")
    scalability_score: Optional[float] = Field(default=0, description="Scalability Score")
    cac_payback_months: Optional[float] = Field(default=0, description="CAC Payback (Months)")
    viral_coefficient: Optional[float] = Field(default=0, description="Viral Coefficient")
    cash_on_hand_usd: Optional[float] = Field(default=0, description="Cash on Hand (USD)")
    brand_strength_score: Optional[float] = Field(default=0, description="Brand Strength Score")
    distribution_partner_count: Optional[float] = Field(default=0, description="Distribution Partner Count")
    ltv_cac_ratio: Optional[float] = Field(default=0, description="LTV/CAC Ratio")
    sales_cycle_length_days: Optional[float] = Field(default=0, description="Sales Cycle (Days)")
    team_size_total: Optional[float] = Field(default=0, description="Team Size (Total)")
    has_debt: Optional[bool] = Field(default=False, description="Has Debt")
    debt_ratio: Optional[float] = Field(default=0, description="Debt Ratio")
    team_diversity_percent: Optional[float] = Field(default=0, description="Team Diversity (%)")
    current_mrr: Optional[float] = Field(default=0, description="Current MRR")
    team_engagement_score: Optional[float] = Field(default=0, description="Team Engagement Score")
    customer_churn_rate_percent: Optional[float] = Field(default=0, description="Customer Churn Rate (%)")
    net_profit_margin_percent: Optional[float] = Field(default=0, description="Net Profit Margin (%)")
    gross_margin_percent: Optional[float] = Field(default=0, description="Gross Margin (%)")
    sam_size_usd: Optional[float] = Field(default=0, description="SAM Size (USD)")
    product_stage: Optional[str] = Field(default='', description="Product Stage")
    burn_multiple: Optional[float] = Field(default=0, description="Burn Multiple")
    data_source: Optional[str] = Field(default='', description="Data Source")
    user_growth_rate_percent: Optional[float] = Field(default=0, description="User Growth Rate (%)")
    nps_score: Optional[float] = Field(default=0, description="NPS Score")
    complementary_skills_index: Optional[float] = Field(default=0, description="Complementary Skills Index")
    revenue_growth_rate: Optional[float] = Field(default=0, description="Revenue Growth Rate")
    employee_turnover_rate_percent: Optional[float] = Field(default=0, description="Employee Turnover Rate (%)")
    competition_intensity: Optional[float] = Field(default=5, description="Competition Intensity")
    market_adoption_stage: Optional[str] = Field(default='', description="Market Adoption Stage")
    switching_cost_score: Optional[float] = Field(default=0, description="Switching Cost Score")
    key_person_dependency: Optional[float] = Field(default=0, description="Key Person Dependency")
    patent_count: Optional[float] = Field(default=0, description="Patent Count")
    product_uptime_percent: Optional[float] = Field(default=99.9, description="Product Uptime (%)")
    security_compliance: Optional[str] = Field(default='', description="Security Compliance")
    customer_concentration_percent: Optional[float] = Field(default=0, description="Customer Concentration (%)")
    success_label: Optional[bool] = Field(default=False, description="Success Label")
    burn_rate_usd: Optional[float] = Field(default=0, description="Burn Rate (USD)")
    waitlist_size: Optional[float] = Field(default=0, description="Waitlist Size")
    has_network_effect: Optional[bool] = Field(default=False, description="Has Network Effect")
    switching_cost: Optional[str] = Field(default='', description="Switching Cost")
    product_retention_30d: Optional[float] = Field(default=0, description="30-Day Retention (%)")
    product_retention_90d: Optional[float] = Field(default=0, description="90-Day Retention (%)")
    mau: Optional[float] = Field(default=0, description="Monthly Active Users")
    moat_barriers: Optional[List[str]] = Field(default=[], description="Moat Barriers")
    linkedin_connections_total: Optional[float] = Field(default=0, description="LinkedIn Connections")
    twitter_followers_total: Optional[float] = Field(default=0, description="Twitter Followers")
    domain_expertise_years_avg: Optional[float] = Field(default=0, description="Domain Expertise (Years)")
    gender_diversity_index: Optional[float] = Field(default=0, description="Gender Diversity Index")
    geography_diversity_index: Optional[float] = Field(default=0, description="Geography Diversity Index")
    founders_count: Optional[float] = Field(default=1, description="Founders Count")
    previous_exits_count: Optional[float] = Field(default=0, description="Previous Exits Count")
    has_patents: Optional[bool] = Field(default=False, description="Has Patents")
    total_funding_usd: Optional[float] = Field(default=0, description="Total Funding (USD)")
    revenue_annual_usd: Optional[float] = Field(default=0, description="Annual Revenue (USD)")
    ltv_usd: Optional[float] = Field(default=0, description="Lifetime Value (USD)")
    regulation_flags: Optional[List[str]] = Field(default=[], description="Regulation Flags")
    top3_competitor_share_pct: Optional[float] = Field(default=0, description="Top-3 Competitor Share (%)")
    roles_present: Optional[List[str]] = Field(default=[], description="Roles Present")
    competitors_named_count: Optional[float] = Field(default=0, description="Competitors Named Count")
    tam_justification: Optional[str] = Field(default='', description="TAM Justification")
    industry_regulation_level: Optional[str] = Field(default='', description="Industry Regulation Level")
    claimed_tam_usd: Optional[float] = Field(default=0, description="Claimed TAM (USD)")
    claimed_cagr_pct: Optional[float] = Field(default=0, description="Claimed CAGR (%)")
    burn_stddev_usd: Optional[float] = Field(default=0, description="Burn StdDev (USD)")
    funding_rounds_count: Optional[float] = Field(default=0, description="Funding Rounds Count")
    committed_funding_usd: Optional[float] = Field(default=0, description="Committed Funding (USD)")
    years_experience_avg: Optional[float] = Field(default=0, description="Avg Years Experience")
    team_size_full_time: Optional[float] = Field(default=0, description="Team Size (Full-Time)")
    degree_level_mode: Optional[str] = Field(default='', description="Degree Level Mode")
    cac_usd: Optional[float] = Field(default=0, description="CAC (USD)")
    previous_exit_max_value_usd: Optional[float] = Field(default=0, description="Prev Exit Max Value (USD)")


class Alert(BaseModel):
    """Model for alerts"""
    type: str
    message: str
    severity: str = "warning"  # info, warning, error


class AnalysisResult(BaseModel):
    """Result of startup analysis"""
    pillar_scores: Dict[str, float]
    overall_score: float
    success_probability: float
    alerts: List[Alert] = []


class RunwaySimulation(BaseModel):
    """Model for runway simulation results"""
    dates: List[str]
    capital: List[float]
    runway_months: int


class PortfolioSimulation(BaseModel):
    """Model for portfolio simulation results"""
    companies: List[str]
    scores: Dict[str, Dict[str, float]]