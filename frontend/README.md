# 🌐 Frontend Service

An ultra-lightweight web server built on `nginx:alpine-slim` serving the Real Estate Intelligence Leaflet UI, charts, and data explorer.

## Features
- Serves static assets (`index.html`, `data.html`, JS, CSS) with gzip compression and caching.
- Reverse-proxies `/api/*` and `/metrics` to the Backend Service.
- Minimal resource footprint (~11MB image size).
- Runs independently on port 80.

## Standalone Build & Run

### 1. Build the Docker image
```bash
docker build -t real-estate-frontend -f frontend/Dockerfile .
```

### 2. Run the container
```bash
docker run -d --name real-estate-frontend \
  -p 80:80 \
  real-estate-frontend
```

When running with a separate backend container or local backend:
- Ensure the backend is reachable via the Nginx upstream host (`backend:8000`) or configure via Docker network / host gateway.

