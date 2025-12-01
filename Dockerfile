# Use python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy all files to the container
COPY . .

# Install the package with "magi" (AI/OCR) dependencies
RUN pip install --no-cache-dir ".[magi]"

# Expose the default Streamlit port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
