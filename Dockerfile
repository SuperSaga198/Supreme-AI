# Fetch high-performance infrastructure baseline image containing core compilation setups
FROM vllm/vllm-openai:latest

WORKDIR /app

# Package project directory requirements files into working sandbox
COPY requirements.txt .
COPY main.py .

# Compile code package dependencies silently
RUN pip install --no-cache-dir -r requirements.txt

# Open the network port container layer
EXPOSE 10000

# Set environment optimization directives
ENV PYTHONUNBUFFERED=1
ENV ENFORCE_EAGER=True

# =================================================================
# CRITICAL FIX: Reset the base image entrypoint override
# =================================================================
# This stops vLLM from hijacking the startup sequence and forcing port 8000.
ENTRYPOINT []

# Execute the native script application loop directly down the wire
CMD ["python", "main.py"]
