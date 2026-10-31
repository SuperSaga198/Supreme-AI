import os
import asyncio
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import modal

# 1. Define the Cloud Container Environment Architecture
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi>=0.110.0",
        "uvicorn>=0.28.0",
        "groq>=0.5.0",
        "pydantic>=2.6.0",
        "vllm==0.6.3",  # Pins robust vLLM version matching standard compilation builds
    )
)

# Initialize Modal App wrapper mapping workspace tracking
app = modal.App("supreme-ai-hybrid-engine")

# Initialize secret environments mapping keys automatically
groq_secret = modal.Secret.from_name("my-groq-secret")

# Define API Request Schema
class SupremeRequest(BaseModel):
    prompt: str
    mode: str = "standard"  # "standard" (Groq Cloud) or "advanced" (On-Demand Local GPU vLLM)
    max_tokens: int = 1024
    temperature: float = 0.7

# =================================================================
# MODULE A: REMOTE ON-DEMAND GPU WORKSPACE LAYER (vLLM Engine)
# =================================================================
@app.cls(
    image=image,
    gpu="A10G:2",          # Automatically provisions 2 physical NVidia A10G GPUs on request!
    timeout=600,           # Generous timeout ceiling for model loading
    concurrency_limit=1,   # Prevents race conditions over physical device locks
)
class SupremeLocalCore:
    @modal.enter()
    def initialize_weights(self):
        """Loads model weights directly into GPU memory only when invoked."""
        from vllm import LLMEngine, EngineArgs
        
        # Point this to where your compiled weights are stored or pulled from
        # e.g., an absolute path, mounted volume, or Hugging Face repo ID
        self.engine_path = "./supreme-compiled-engine"
        
        if os.path.exists(self.engine_path):
            print(f"[INFO] Initializing TensorRT-LLM workspace across active GPUs...")
            engine_args = EngineArgs(
                model=self.engine_path,
                tensor_parallel_size=2,  # Safely binds to the 2 allocated GPUs mapped above
                block_size=16,
                gpu_memory_utilization=0.90,
                enforce_eager=True
            )
            self.engine_instance = LLMEngine.from_engine_args(engine_args)
            print("[INFO] Advanced GPU Engine core successfully loaded.")
        else:
            self.engine_instance = None
            print(f"[WARNING] Model weights directory missing at {self.engine_path}.")

    @modal.method()
    async def generate_stream(self, prompt: str, max_tokens: int, temperature: float):
        """Generates tokens directly out of GPU VRAM workspace layers."""
        if not self.engine_instance:
            yield "[System Error]: Local GPU Engine is missing compiled weights."
            return

        from vllm import SamplingParams
        from vllm.utils import random_uuid
        
        request_id = random_uuid()
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95
        )
        
        results_generator = self.engine_instance.generate(prompt, sampling_params, request_id)
        async for request_output in results_generator:
            if request_output.outputs:
                yield request_output.outputs[-1].text
            await asyncio.sleep(0)

# =================================================================
# MODULE B: WEB ROUTER LAYERS (FastAPI Public Web Gateway)
# =================================================================
@app.function(
    image=image, 
    secrets=[groq_secret],
    allow_cross_origin_all=True  # Enables CORS for frontend traffic channels
)
@modal.asgi_app()
def web_gateway():
    fastapi_app = FastAPI(title="Supreme AI Hybrid Routing Engine", version="2.0.0")

    # Native internal helper for Groq cloud connections
    def get_groq_client():
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None
        from groq import AsyncGroq
        return AsyncGroq(api_key=api_key)

    @fastapi_app.post("/v1/supreme/chat")
    async def process_hybrid_route(request: SupremeRequest):
        # --- MODE A: ADVANCED CORE ---
        if request.mode == "advanced":
            # Spin up the remote GPU class infrastructure on demand dynamically
            try:
                gpu_core = SupremeLocalCore()
                # Returns a streaming generator across network interfaces
                async def remote_gpu_iterator():
                    async for token in gpu_core.generate_stream.remote_gen.aio(
                        request.prompt, request.max_tokens, request.temperature
                    ):
                        yield token
                return StreamingResponse(remote_gpu_iterator(), media_type="text/event-stream")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"On-Demand GPU provisioning failed: {str(e)}")

        # --- MODE B: STANDARD ROUTE (GROQ CLOUD) ---
        else:
            groq_client = get_groq_client()
            if not groq_client:
                raise HTTPException(status_code=503, detail="Groq API key environment missing in Modal secrets dashboard.")

            async def stream_groq_api() -> AsyncGenerator[str, None]:
                try:
                    completion = await groq_client.chat.completions.create(
                        model="llama3-70b-8192",  
                        messages=[{"role": "user", "content": request.prompt}],
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        stream=True,
                    )
                    async for chunk in completion:
                        delta = chunk.choices.delta.content
                        if delta:
                            yield delta
                except Exception as e:
                    yield f"\n[Groq Channel Error]: {str(e)}"

            return StreamingResponse(stream_groq_api(), media_type="text/event-stream")

    @fastapi_app.get("/health")
    async def health_check():
        return {"status": "active", "gateway": "operational"}

    return fastapi_app
