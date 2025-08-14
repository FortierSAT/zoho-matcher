#!/usr/bin/env bash
set -euo pipefail

# Gunicorn binds to $PORT on Render
: "${PORT:=10000}"
exec gunicorn app:app --bind "0.0.0.0:${PORT}" --workers 2 --threads 8 --timeout 120
