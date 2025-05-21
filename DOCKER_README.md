# FlashDNA Docker Deployment

This guide explains how to deploy the FlashDNA application using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose installed on your system

## Quick Start

1. **Clone the repository**

```bash
git clone <repository-url>
cd FLASH
```

2. **Set up environment variables**

Copy the example environment file:

```bash
cp env.example .env
```

Edit the `.env` file to set your desired configuration, especially:
- `SECRET_KEY`: Set this to a secure random string
- `DATABASE_URL`: Modify if using a different database than SQLite
- `CORS_ORIGINS`: Update with your frontend URL(s) if needed

3. **Build and start the containers**

```bash
docker-compose up -d --build
```

4. **Verify the application is running**

```bash
curl http://localhost:8000/
```

You should see a JSON response indicating the API is healthy.

## Configuration Options

### Changing the Port

By default, the API runs on port 8000. To change this:

1. Update the `PORT` value in your `.env` file
2. Also update the port mapping in `docker-compose.yml`:
   ```yaml
   ports:
     - "YOUR_PORT:8000"
   ```

### Using PostgreSQL

To use PostgreSQL instead of SQLite:

1. Uncomment the `db` service and `volumes` section in `docker-compose.yml`
2. Update the `DATABASE_URL` in your `.env` file to point to the PostgreSQL service:
   ```
   DATABASE_URL=postgresql://postgres:postgres@db:5432/flashcamp
   ```

## Volumes and Data Persistence

The Docker Compose configuration mounts several directories to persist data:

- `./data:/app/data`: Application data
- `./models:/app/models`: ML models
- `./reports:/app/reports`: Generated reports
- `./logs:/app/logs`: Application logs
- `./tmp:/app/tmp`: Temporary files

Make sure these directories exist and have appropriate permissions.

## Troubleshooting

### Database Connection Issues

If you encounter database connection issues:

1. Check your `DATABASE_URL` is correctly set
2. Ensure the database service is running (if using PostgreSQL)
3. Check the logs with `docker-compose logs api`

### Container Won't Start

If the API container fails to start:

1. Check the logs: `docker-compose logs api`
2. Ensure all required environment variables are set
3. Verify that all required directories exist

## Health Checks

The API service includes a health check that runs every 30 seconds. You can check the health status with:

```bash
docker ps
```

Look for the `STATUS` column to see if the container is `healthy` or `unhealthy`.

## Production Deployment

For production deployment, additional configuration is recommended:

1. Set `ENVIRONMENT=production` in your `.env` file
2. Use a proper database like PostgreSQL instead of SQLite
3. Set up a reverse proxy (like Nginx) in front of the API
4. Implement proper security measures (HTTPS, proper firewall rules, etc.)
5. Set up monitoring and alerting 