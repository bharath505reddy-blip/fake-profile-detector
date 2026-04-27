FROM python:3.11-slim

# System deps for matplotlib, scientific libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads static/charts models instance

# Environment
ENV FLASK_APP=app.py
ENV FLASK_DEBUG=0
ENV PYTHONUNBUFFERED=1

EXPOSE $PORT

CMD python -m gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 --timeout 120 app:app
