FROM python:3.12-slim

# Prevent Python from writing .pyc files & ensure stdout flush
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (optional but handy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl tini && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app.py matcher.py start.sh ./
RUN chmod +x /app/start.sh

# Use tini as PID 1 for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Render injects $PORT. We just run start.sh (gunicorn binds to $PORT)
CMD ["./start.sh"]
