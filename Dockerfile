# Use official vLLM image which comes pre-packaged with TensorRT-LLM and CUDA drivers
FROM vllm/vllm-openai:latest

WORKDIR /app

# Copy configuration files and server script into the container
COPY requirements.txt .
COPY main.py .

# Install explicit system dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Open network port 8000 for server traffic
EXPOSE 8000

# Set environment variables to optimize vLLM behavior inside a serverless handler
ENV PYTHONUNBUFFERED=1
ENV ENFORCE_EAGER=True

# Execute the FastAPI server on launch
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
