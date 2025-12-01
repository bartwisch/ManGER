#!/bin/bash
set -e

cd /app

# Pull latest code from GitHub
echo "📥 Pulling latest code from GitHub..."
if [ -d ".git" ]; then
    git pull origin main
else
    git clone https://github.com/bartwisch/ManGER.git .
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
