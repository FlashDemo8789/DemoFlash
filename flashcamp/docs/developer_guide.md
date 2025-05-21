# FlashCAMP Developer Guide

This guide provides detailed information for developers working on the FlashCAMP project.

## Architecture Overview

FlashCAMP follows a modern web application architecture with a clear separation of concerns:

1. **Frontend**: React Single Page Application with TypeScript
2. **Backend**: FastAPI REST API with Python
3. **Database**: SQL database (SQLite in development, PostgreSQL in production)
4. **Monitoring**: Prometheus and Grafana for metrics and visualization

### Component Interaction

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

## Development Environment Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker and Docker Compose
- Git

### Local Setup

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

3. **Set up environment**:
   ```bash
   cp env.template .env
   # Edit .env with your configuration
   ```

4. **Initialize the database**:
   ```bash
   python scripts/init_db.py
   ```

5. **Train models** (if not already trained):
   ```bash
   python pipelines/train_baseline.py --data data/gold/v1.parquet --models models/
   ```

6. **Set up frontend**:
   ```bash
   cd frontend
   npm install
   cd ..
   ```

## Development Workflow

### Backend Development

#### Running the Backend Server

```bash
# From the project root
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

The API documentation will be available at:
- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

#### Creating Database Migrations

When you modify the database models, you need to create a new migration:

```bash
alembic revision --autogenerate -m "Description of changes"
```

Apply migrations:

```bash
alembic upgrade head
```

#### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_api.py

# Run with coverage report
pytest --cov=backend
```

### Frontend Development

#### Running the Frontend Development Server

```bash
# From the frontend directory
npm run dev
```

The frontend will be available at http://localhost:3000.

#### Building for Production

```bash
# From the frontend directory
npm run build
```

#### Running Frontend Tests

```bash
# From the frontend directory
npm test
```

## Project Structure

### Backend Structure

- **backend/app.py**: Main FastAPI application
- **backend/config.py**: Configuration settings
- **backend/database.py**: Database models and connection
- **backend/schemas.py**: Pydantic schemas for data validation
- **backend/validation.py**: Advanced input validation
- **backend/logging_config.py**: Structured logging configuration
- **backend/metrics.py**: Prometheus metrics collection
- **backend/app/**: Application components:
  - **engines/**: Analysis engines and prediction models
- **backend/schema/**: JSON Schema definitions
- **backend/sims/**: Simulation modules

### Frontend Structure

- **frontend/src/**: Source code
  - **components/**: Reusable UI components
  - **pages/**: Application pages
  - **hooks/**: Custom React hooks
  - **services/**: API service clients
  - **utils/**: Utility functions
  - **types/**: TypeScript type definitions
- **frontend/public/**: Static assets

## API Reference

### Core Endpoints

#### `GET /`
Health check endpoint that returns API status.

#### `POST /api/analyze`
Analyzes startup metrics and returns scores.

**Request Body**: `MetricsInput` schema
**Response**: Pillar scores, overall score, success probability, and alerts

#### `POST /api/generate_report`
Generates a PDF report for a startup analysis.

**Request Body**: `MetricsInput` schema
**Response**: PDF file

## Monitoring and Observability

### Logging

The application uses structured logging with the following log levels:
- `DEBUG`: Detailed debugging information
- `INFO`: General information about application operation
- `WARNING`: Warning events that might require attention
- `ERROR`: Error events that might still allow the application to continue
- `CRITICAL`: Critical events that may lead to application failure

Logs are written to:
- Console (all environments)
- JSON-formatted log files in `logs/` directory:
  - `flashcamp.log`: All logs (INFO and above)
  - `error.log`: Error logs (ERROR and above)

Each log entry includes:
- Timestamp
- Log level
- Module and function
- Request ID for request correlation
- Structured data for machine processing

### Metrics

The application collects the following metrics:
- **Request metrics**: Count, latency by endpoint
- **Business metrics**: Analyses performed, reports generated
- **System metrics**: Active requests, model prediction time
- **Error metrics**: Error counts by type

Metrics are available at the `/metrics` endpoint in Prometheus format.

## Code Style and Conventions

### Python

- Follow PEP 8 style guide
- Use type hints for all function parameters and return values
- Document all functions with docstrings
- Organize imports in alphabetical order (standard, third-party, local)
- Use f-strings for string formatting
- Use meaningful variable names

### TypeScript

- Follow Airbnb style guide
- Use TypeScript interfaces for data structures
- Use functional components with hooks for React
- Avoid any types where possible
- Use async/await instead of Promises

## Deployment

### Docker Deployment

Build and run with Docker Compose:

```bash
docker-compose up -d
```

### Production Deployment Checklist

Before deploying to production:

1. Update the `.env` file with production settings
2. Set a strong `SECRET_KEY`
3. Configure database connection for PostgreSQL
4. Set `ENVIRONMENT=production`
5. Set up proper CORS origins
6. Run database migrations
7. Build optimized frontend assets

## Troubleshooting

### Common Issues

#### Database Connection Issues

- Check that the database exists and is accessible
- Verify that the `DATABASE_URL` is correctly set
- For PostgreSQL, check that the user has the correct permissions

#### Frontend API Connection Issues

- Check that the backend API is running
- Verify that CORS is properly configured
- Check browser console for error messages

## Contributing

### Pull Request Process

1. Create a new branch for your feature or bug fix
2. Make your changes with appropriate tests
3. Run all tests to ensure they pass
4. Update documentation as needed
5. Submit a pull request with a clear description of changes

### Code Review Guidelines

- Code should follow the project's style guides
- All tests should pass
- Documentation should be updated
- Code should be maintainable and well-structured 