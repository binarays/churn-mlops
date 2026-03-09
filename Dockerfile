# -----------------------------
# Base Image
# -----------------------------
FROM python:3.9-slim

# -----------------------------
# Set working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Install system dependencies
# -----------------------------
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    dnsutils \
    iputils-ping \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Copy requirements file first
# (for better Docker caching)
# -----------------------------
COPY requirements.txt .

# -----------------------------
# Install Python dependencies
# -----------------------------
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy project files
# -----------------------------
COPY . .

# -----------------------------
# Expose API port
# -----------------------------
EXPOSE 7860

# -----------------------------
# Start FastAPI Server (Production Ready)
# -----------------------------
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]