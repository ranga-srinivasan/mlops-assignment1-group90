# Lightweight python base image
FROM python:3.10-slim

WORKDIR /app

# Copy dependency file first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy rest of project
COPY . /app

# Expose FastAPI port
EXPOSE 8000

# Start API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
