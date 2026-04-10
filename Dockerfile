FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY *.py ./
COPY .env.example .env.example

# Create data directories
RUN mkdir -p ai_data

# Default command
CMD ["python", "deepalpha.py"]
