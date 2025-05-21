# FLASH Project Technical Documentation

## Project Overview
FLASH is a machine learning platform designed to streamline the development, deployment, and monitoring of ML models. The project consists of a backend service built with FastAPI, a frontend interface, and monitoring capabilities with Prometheus and Grafana.

## System Architecture

### Components
- **Backend**: Python-based FastAPI application with PostgreSQL/SQLite database
- **Frontend**: Web interface for interacting with the platform
- **Monitoring**: Prometheus for metrics collection and Grafana for visualization
- **Database**: SQL database with SQLAlchemy ORM and Alembic migrations

### Infrastructure
The application is containerized using Docker with services defined in `docker-compose.yml`:
- Backend service (port 8000)
- Frontend service (port 3000)
- Prometheus (port 9090)
- Grafana (port 3001)

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
cd flashcamp
```

2. Set up environment
```bash
cp env.template .env
# Edit .env with your configuration
```

3. Initialize the database
```bash
# First, install the package in development mode
pip install -e .

# Create initial migration (only needed if migration files don't exist)
python -m alembic revision --autogenerate -m "Initial migration"

# Apply migrations to create database tables
python -m alembic upgrade head
```

4. Start the services
```bash
docker-compose up -d
```

5. Access the application
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Grafana: http://localhost:3001
- Prometheus: http://localhost:9090

## Development Guidelines

### Backend Development
The backend is built with FastAPI and follows these conventions:
- Code is organized in the `backend/` directory
- API endpoints are defined in `backend/app.py`
- Data schemas are defined in `backend/schemas.py`
- Database models are defined in `backend/schema/models.py`
- Database operations (CRUD) are in `backend/schema/crud.py`
- Database connection is managed in `backend/schema/database.py`
- Database migration scripts are in `migrations/`
- Environment configuration is managed via `.env` file

### Database Schema
The database schema is implemented using SQLAlchemy ORM with the following models:
- **Startup**: Core entity representing a startup with basic information and metrics data
- **Analysis**: Records of analyses performed on startups with pillar scores and success probabilities
- **Report**: Generated PDF reports for startups
- **User**: User accounts for the application

The schema structure is organized as follows:
- `backend/schema/models.py`: SQLAlchemy ORM model definitions
- `backend/schema/crud.py`: CRUD operations for all models
- `backend/schema/database.py`: Database connection management
- `backend/schema/__init__.py`: Module exports

### Frontend Development
The frontend follows modern web development practices:
- Code is organized in the `frontend/` directory
- Components are in `frontend/src/components/`
- Pages are in `frontend/src/pages/`

## Data Pipeline
The data flow in the system follows this pattern:
1. Raw data ingestion into `data/` directory
2. Feature extraction using pipelines in `pipelines/`
3. Model training in `models/`
4. Analysis and reporting in `reports/`

## Monitoring and Observability
The system includes comprehensive monitoring and observability:
- Prometheus collects metrics from the backend service
- Grafana dashboards visualize system performance and model metrics
- Custom business metrics track system usage
- Structured logging with request correlation IDs
- Logs are stored in the `logs/` directory

## Documentation
- `README.md`: Overview of the project (this file)
- `TECHNICAL_DOCUMENTATION.md`: Detailed technical documentation
- `docs/developer_guide.md`: Guide for developers
- `env.template`: Template for environment configuration

## Testing
- Unit tests are in `tests/` directory
- Run tests with `pytest`
- API tests are available in `tests/test_api.py`

## Recent Changes
- Implemented complete database schema with SQLAlchemy models (Startup, Analysis, Report, User)
- Added CRUD operations for all database models
- Added database connection management module
- Added curl to Dockerfile.backend for health checks
- Added developer guide
- Fixed CORS configuration issues

## License
[License information]

## Contract-Model Alignment

The system uses `contracts/metrics_full.json` as the single source of truth for the data contract. This file contains the exact 79 features expected by the XGBoost success prediction model.

### Key Components

1. **Metrics Contract**: `contracts/metrics_full.json` defines all metrics with their types and requirements.
2. **Feature Map**: Auto-generated `backend/feature_map.py` exports the `FEATURES` list that ensures exact alignment with the model.
3. **Feature Preparation**: `backend/features.py` handles automatic padding and one-hot encoding.
4. **CI Guard**: A test in CI/CD verifies model and contract alignment to prevent drift.

### Maintaining Alignment

When making changes:

1. Edit `contracts/metrics_full.json` directly if you need to modify the schema.
2. Run `python flashcamp/generators/build_schema.py --contract flashcamp/contracts/metrics_full.json` to regenerate all dependent files.
3. Restart the service (`find flashcamp -name '*.pyc' -delete` to clear caches first).

The CI pipeline will catch any misalignment between model features and schema. 