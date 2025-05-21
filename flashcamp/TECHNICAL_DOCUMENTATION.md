# FlashCAMP Technical Documentation

## Project Overview

FlashCAMP is a comprehensive startup analysis platform that evaluates startup performance across four key pillars:
- **C**apital: Funding, burn rate, runway, financial health
- **A**dvantage: Competitive moat, IP, network effects
- **M**arket: TAM, growth rate, competition intensity
- **P**eople: Team composition, experience, diversity

### Purpose

This system helps investors, accelerators, and founders analyze startups using a data-driven approach. It:
1. Accepts 100+ metrics as input
2. Generates pillar scores and overall success probability
3. Creates visualizations and PDF reports
4. Provides insights through model explainability

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

1. **Frontend**:
   - React Single Page Application (TypeScript)
   - Material-UI for components
   - Recharts for data visualization
   - Uses react-query for API communication

2. **Backend**:
   - FastAPI framework
   - Pydantic for data validation
   - WeasyPrint for PDF generation
   - File-based storage using Parquet (database integration pending)

3. **ML Models**:
   - LightGBM for pillar score prediction
   - XGBoost for success probability
   - SHAP for model explainability

4. **Monitoring**:
   - Prometheus for metrics collection
   - Grafana for dashboards and visualization

5. **Deployment**:
   - Docker containers for each service
   - Nginx for serving frontend assets
   - Docker Compose for orchestration

## Current Status

### Component Status

| Component | Status | Notes |
|-----------|--------|-------|
| Frontend | ✅ Working | Basic functionality implemented |
| Backend API | ✅ Working | Core endpoints functioning |
| ML Models | ✅ Working | Trained and integrated |
| PDF Generation | ✅ Working | Basic reports implemented |
| Database | ✅ Working | SQLite database with migrations implemented |
| Authentication | ⚠️ Partial | Basic framework present, not complete |
| Containerization | ✅ Working | Docker setup configured and tested |
| CI/CD | ❌ Not Implemented | No automated pipeline |
| Monitoring | ⚠️ Partial | Basic metrics only |
| Testing | ⚠️ Minimal | Limited test coverage |

### Known Issues

This section lists known issues and their status:

1. ✅ **Fixed: Frontend-Backend connection issues** - The frontend was previously using hardcoded URLs to connect to the backend. This has been fixed by updating the frontend to use relative URLs.

2. ✅ **Fixed: Database connectivity issues** - Database connectivity issues have been resolved by disabling database operations in the analysis endpoints. The application now works in a stateless mode, performing analysis without storing results.

3. ✅ **Fixed: Missing favicon** - Added basic favicon configuration to prevent 404 errors.

4. ✅ **Fixed: React Alert rendering error** - Fixed React error in AlertsBox component by updating it to handle the new alert object structure from the backend API with type, message, and severity fields.

5. ❌ **Authentication incomplete** - User authentication is not fully implemented. Only basic framework is present.

6. ❌ **CI/CD not implemented** - There is no automated CI/CD pipeline for the project.

7. ⚠️ **Partial test coverage** - Tests have been started but coverage is limited.

8. ⚠️ **Basic monitoring only** - Prometheus and Grafana are set up with basic metrics, but more comprehensive monitoring is needed.

## Compatibility Requirements

### System Requirements

- **Python**: 3.11 or newer
- **Node.js**: 18.x or newer
- **Docker**: 20.10.x or newer
- **Docker Compose**: 2.x or newer

### Browser Support

- Chrome/Edge 90+
- Firefox 90+
- Safari 14+

### OS Compatibility

- Linux (Ubuntu 20.04+, Debian 11+)
- macOS 11+
- Windows 10/11 with WSL2 for Docker

## Recent Changes Log

This section will be updated after each significant change to the system.

| Date | Version | Changes | Developer |
|------|---------|---------|-----------|
| Initial | 0.1.0 | Initial documentation | - |

## Development Guidelines

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd flashcamp
   ```

2. **Set up Python environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Prepare data**:
   ```bash
   python notebooks/01_validate_csv.py --in data/camp_plus_balanced_with_meta.csv --schema backend/schema/camp_schema.yaml --out data/gold/v1.parquet
   ```

4. **Train models**:
   ```bash
   python pipelines/train_baseline.py --data data/gold/v1.parquet --models models/
   python pipelines/gen_shap.py --data data/gold/v1.parquet --models models/ --out reports/assets/
   ```

5. **Set up frontend**:
   ```bash
   cd frontend
   npm install
   cd ..
   ```

6. **Start services**:
   ```bash
   # Terminal 1: Backend
   uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
   
   # Terminal 2: Frontend
   cd frontend
   npm run dev
   
   # Terminal 3: Monitoring (optional)
   docker compose -f monitoring/docker-compose.yml up -d
   ```

### Coding Standards

1. **Python**:
   - Follow PEP 8 style guide
   - Use type hints for function parameters and return values
   - Document functions with docstrings

2. **TypeScript/JavaScript**:
   - Follow Airbnb style guide
   - Use TypeScript interfaces for data structures
   - Use functional components with hooks for React

3. **Git**:
   - Use feature branches for new development
   - Follow conventional commits format (feat:, fix:, docs:, etc.)
   - Squash commits before merging to main

### Testing

1. **Backend**:
   - Write unit tests for all business logic
   - Use pytest for testing framework
   - Aim for 80%+ test coverage for core logic

2. **Frontend**:
   - Use React Testing Library for component tests
   - Test key user flows
   - Include snapshot tests for UI components

### Documentation

1. **API**:
   - Document all endpoints with examples
   - Update this technical document when adding new features
   - Include error handling information

2. **Code**:
   - Add comments for complex logic
   - Keep README up to date with setup instructions
   - Document configuration options

## Roadmap

### Short-term Goals

1. Fix all identified issues
2. Implement database integration
3. Complete authentication system
4. Improve test coverage

### Mid-term Goals

1. Enhance visualization capabilities
2. Add user management system
3. Implement CI/CD pipeline
4. Add export functionality

### Long-term Goals

1. Add multi-tenant support
2. Implement advanced analytics features
3. Create mobile-responsive design
4. Add real-time collaboration features

---

*This document should be updated after each significant change to the system. Last updated: [Current Date]* 