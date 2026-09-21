# Use a lightweight Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

RUN sed -i 's|http://|https://|g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        wget \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy only requirements first (for caching efficiency)
COPY requirements.txt .

# Install all dependencies (except faiss-gpu)
RUN grep -v "faiss-gpu" requirements.txt > temp-req.txt && pip install --no-cache-dir -r temp-req.txt

# Install FAISS (CPU-only, works with Python 3.12)
RUN pip install faiss-cpu

# Copy rest of the app
COPY . .

# Expose Flask port
EXPOSE 5000

# Start the Flask app with Waitress (production WSGI server)
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "--threads=4", "app:app"]
