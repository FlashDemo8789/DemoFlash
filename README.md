# FLASH Project

## Recent Updates

**July 5, 2024**: Implemented new hierarchical model architecture (v2):

1. **Specialized Pillar Models**: Created four dedicated LightGBM models for each pillar (Capital, Advantage, Market, People).
2. **Meta-Model Ensemble**: Implemented XGBoost meta-model that combines pillar outputs for final prediction.
3. **Improved Explainability**: Enhanced understanding of predictions with pillar-specific explanations.
4. **Better Performance**: Boosted prediction metrics with AUC increasing from ~0.78 to 0.85+.
5. **Added Training Pipeline**: Created `train_hierarchical_models.sh` for easy model retraining.

**June 30, 2024**: Fixed module import issues and improved model loading functionality:

1. **Fixed module import errors**: Created a run.sh script that properly sets PYTHONPATH to resolve import issues.
2. **Added model loading metrics**: Implemented MODEL_LOADED counter in Prometheus to track successful model loads.
3. **Fixed circular import**: Resolved circular dependency in backend/app/__init__.py.
4. **Added verification script**: Created verify_metrics.py to test model loading and Prometheus metrics.

**June 26, 2024**: Fixed analysis consistency issues that were causing all analyses to look identical:

1. **Fixed model loading**: Updated model path to point to the correct location in `/mnt/data/`.
2. **Re-enabled validation**: Enhanced input validation to catch inconsistent or invalid inputs.
3. **Fixed number input handling**: Improved numeric field handling in the frontend to properly process decimal values.
4. **Added consistency testing**: Created a test to verify that different inputs produce different analysis results.

These fixes ensure that different startup inputs now produce appropriately different analysis results.

## Project Overview
FLASH (FlashCAMP) is a comprehensive startup analysis platform that evaluates startup performance across four key pillars:
- **C**apital: Funding, burn rate, runway, financial health
- **A**dvantage: Competitive moat, IP, network effects
- **M**arket: TAM, growth rate, competition intensity
- **P**eople: Team composition, experience, diversity

The system helps investors, accelerators, and founders analyze startups using a data-driven approach by:
1. Processing 100+ metrics as input
2. Generating pillar scores and overall success probability
3. Creating visualizations and PDF reports
4. Providing insights through model explainability

## Hierarchical Model Architecture

The platform uses a hierarchical ensemble model architecture to predict startup success probability:

```
                       ┌───────────────────┐
                       │ XGBoost Meta-Model│
                       │(success_xgb.joblib)│
                       └─────────┬─────────┘
                                 │
            ┌──────────┬─────────┼─────────┬──────────┐
            │          │         │         │          │
   ┌────────▼─────┐┌───▼────────┐┌─────────▼──┐┌──────▼───────┐
   │ Capital Model││Advantage Mod││Market Model ││ People Model │
   │(capital_lgbm)││(advantage_lg││(market_lgbm)││(people_lgbm) │
   └──────────────┘└─────────────┘└────────────┘└──────────────┘
```

Each pillar model specializes in a specific aspect of startup success:
- **Capital**: Financial health and fundraising metrics
- **Advantage**: Product differentiation and competitive moat
- **Market**: Market opportunity, growth, and competition
- **People**: Team composition, experience, and leadership

The meta-model combines the outputs of these four pillar models to make the final prediction.

See the [examples directory](/examples) for usage examples and the [models/v2 documentation](/models/v2/README.md) for more details.

## System Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Frontend   │────▶│   Backend   │────▶│   Models    │
│  React SPA  │◀────│  FastAPI    │◀────│  ML Logic   │
└─────────────┘     └─────────────┘     └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │ Monitoring  │
                    │ Prometheus  │
                    │  + Grafana  │
                    └─────────────┘
```

### Components
1. **Frontend**: React Single Page Application with TypeScript and Material-UI
2. **Backend**: FastAPI framework with Pydantic for data validation
3. **ML Models**: LightGBM and XGBoost for predictions, SHAP for model explainability
4. **Monitoring**: Prometheus for metrics collection, Grafana for dashboards
5. **Deployment**: Docker containers for each service

## Setup and Installation

### Prerequisites
- Docker and Docker Compose
- Git
- Python 3.11+ (for local development)
- Node.js 18+ (for local development)

### Getting Started

1. Clone the repository
```bash
git clone <repository-url>
cd FLASH
```

2. Start with Docker Compose
```bash
cd flashcamp
docker-compose up -d
```

3. Access the application
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Grafana: http://localhost:3001
- Prometheus: http://localhost:9090

### Local Development Setup

1. Set up Python environment
```bash
cd flashcamp
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .  # Install the package in editable mode
```

2. Set up frontend
```bash
cd frontend
npm install
```

3. Start services locally
```bash
# Option 1: Using the run script (recommended)
./run.sh

# Option 2: Manual startup
# Terminal 1: Backend
export FLASHDNA_MODEL=$(pwd)/models/success_xgb.joblib
uvicorn flashcamp.backend.app:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Frontend
cd frontend
npm run dev
```

4. Verify setup (optional)
```bash
# Run the verification script to check model loading and metrics
python verify_metrics.py

# Run the test suite
export FLASHDNA_MODEL=$(pwd)/models/success_xgb.joblib
python -m pytest tests/test_analysis_variation.py -v
```

## Directory Structure

- **backend/**: FastAPI application code
  - **app/**: Application logic and endpoints
  - **schema/**: Data validation schemas
  - **contracts/**: API interfaces and types
- **frontend/**: React application
  - **src/components/**: Reusable UI components
  - **src/pages/**: Application pages
  - **src/types/**: TypeScript type definitions
- **monitoring/**: Prometheus and Grafana configuration
  - **dashboards/**: Grafana dashboard templates
  - **datasources/**: Grafana data source configuration
- **models/**: Machine learning model files
- **data/**: Data storage
  - **gold/**: Processed data ready for use
- **reports/**: Report templates and assets
- **pipelines/**: Data processing and model training pipelines
- **notebooks/**: Jupyter notebooks for exploration
- **scripts/**: Utility scripts

## Development Guidelines

### Backend Development
- Follow PEP 8 style guide
- Use type hints for function parameters and return values
- Document functions with docstrings

### Frontend Development
- Follow Airbnb style guide for TypeScript
- Use TypeScript interfaces for data structures
- Use functional components with React hooks

### Testing
- Write unit tests for all business logic
- Use pytest for backend testing
- Use React Testing Library for frontend tests

## Data Pipeline
1. Raw data ingestion via CSV files
2. Data validation and transformation
3. Feature extraction
4. Model training and evaluation
5. Visualization and reporting

## Data Quality Tools

### Metrics Cleanup

We've implemented tools to clean up duplicate metrics in the dataset. The main dataset file `camp_plus_balanced_with_meta.csv` contained several duplicate columns with slightly different naming conventions. These duplicates have been standardized to a canonical naming system.

Key cleanup features:
- Script to identify and clean up duplicate metrics (`scripts/cleanup_duplicate_metrics.py`)
- Unit tests to verify cleanup and detect future duplicates (`tests/test_metrics_collinearity.py`)
- GitHub workflow to check for collinearity in PRs

#### Canonical Metric Names

We've standardized on the following canonical names for metrics (removing their aliases):

| Pillar             | Canonical Name                | Previous Aliases                       |
| ------------------ | ----------------------------- | -------------------------------------- |
| **Advantage**      | `patent_count`                | `patents_count`                        |
|                    | `has_network_effect`          | `network_effects_present`              |
| **Market**         | `nps_score`                   | `nps`                                  |
|                    | `burn_rate_usd`               | `monthly_burn_usd`                     |
| **Capital**        | `total_funding_usd`           | `total_capital_raised_usd`             |
|                    | `revenue_annual_usd`          | `annual_revenue_run_rate`              |
| **People**         | `founders_count`              | `founding_team_size`                   |
|                    | `domain_expertise_years_avg`  | `founder_domain_experience_years`      |
|                    | `previous_exits_count`        | `prior_successful_exits_count`         |
| **Info / Context** | `sector`                      | `industry`                             |

#### Running the Cleanup Script

To clean up duplicate metrics in your local environment:

```bash
python scripts/cleanup_duplicate_metrics.py
```

This will:
1. Read the dataset with duplicate metrics
2. Map duplicate columns to their canonical names
3. Write a clean version of the dataset
4. Update metrics JSON files to remove duplicate definitions
5. Test for any remaining collinearity

## Known Issues
- Docker healthcheck configuration requires curl, which has been added to the Dockerfile.backend

## Recent Changes
- Added curl installation to Dockerfile.backend to support container health checks
- Added necessary WeasyPrint dependencies to Dockerfile.backend

## License
[License information] 