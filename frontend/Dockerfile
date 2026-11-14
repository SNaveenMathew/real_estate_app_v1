# ==============================================================================
# Frontend Service Dockerfile (Lightweight Nginx Runner)
# ==============================================================================
FROM nginx:alpine-slim AS runner

# Remove default nginx welcome page
RUN rm -rf /usr/share/nginx/html/* /etc/nginx/conf.d/*

# Copy custom Nginx reverse proxy & static serving configuration
COPY frontend/nginx.conf /etc/nginx/nginx.conf

# Copy static frontend assets (HTML, CSS, JS)
COPY static/index.html /usr/share/nginx/html/index.html
COPY static/data.html /usr/share/nginx/html/data.html
COPY static/ /usr/share/nginx/html/static/

# Expose HTTP port
EXPOSE 80

# Health check using busybox wget (zero extra package overhead)
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD wget -qO- http://127.0.0.1/healthz || exit 1

# Start Nginx in foreground
CMD ["nginx", "-g", "daemon off;"]

