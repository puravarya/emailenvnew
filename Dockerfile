FROM python:3.10-slim

WORKDIR /app

# Install all dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY . .

EXPOSE 7860

# Run server from root so all imports resolve correctly
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]