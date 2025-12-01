#!/bin/bash
set -e

# Pull latest code from GitHub
echo "📥 Pulling latest code from GitHub..."
if [ -d "/app/.git" ]; then
    cd /app
    git pull origin main
else
    rm -rf /app
    git clone https://github.com/bartwisch/ManGER.git /app
    cd /app
fi

# Install the package
pip install -e . --quiet

echo "🚀 Starting ManGER..."
exec streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.maxUploadSize=500
