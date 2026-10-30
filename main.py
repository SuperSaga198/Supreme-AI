import os
import json
import asyncio
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Try importing serverless MLOps dependencies safely
try:
    from vllm import LLMEngine, EngineArgs, SamplingParams
    from vllm.utils import random_uuid
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

try:
    from groq import AsyncGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Initialize FastAPI Application
app = FastAPI(title="Supreme AI Hybrid Routing Engine", version="2.0.0")

# Enable CORS for frontend interface communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define API Request Schema
class SupremeRequest(BaseModel):
    prompt: str
    mode: str = "standard"  # options: "standard" (Groq) or "advanced" (TensorRT-LLM)
    max_tokens: int = 1024
    temperature: float = 0.7

# Global engine placeholders for the serverless container lifecycle
tensorrt_engine_instance = None
groq_client = None

@app.on_event("startup")
async def initialize_system_channels():
    """Triggered once when the serverless container spins up."""
    global tensorrt_engine_instance, groq_client
    
    # 1. Initialize Groq Channel (Standard Mode)
    if GROQ_AVAILABLE:
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            groq_client = AsyncGroq(api_key=api_key)
            print("[INFO] Groq Channel successfully initialized.")
        else:
            print("[WARNING] GROQ_API_KEY not found in environment variables.")
    else:
        print("[WARNING] groq package missing. Standard channel unavailable.")

    # 2. Initialize vLLM + TensorRT-LLM Channel (Advanced Mode)
    if VLLM_AVAILABLE:
        engine_path = "./supreme-compiled-engine"
        if os.path.exists(engine_path):
            print(f"[INFO] Compiling and loading TensorRT engine from {engine_path}...")
            engine_args = EngineArgs(
                model=engine_path,
                tensor_parallel_size=2,       # Splitting across dual-GPU infrastructure
                block_size=16,                # Memory block optimization for PagedAttention
                gpu_memory_utilization=0.90,  # Pre-allocates VRAM for continuous batching
                enforce_eager=True            # Disables overhead graph capture for serverless speed
            )
            tensorrt_engine_instance = LLMEngine.from_engine_args(engine_args)
            print("[INFO] Advanced TensorRT-LLM Core successfully initialized.")
        else:
            print(f"[WARNING] TensorRT engine directory not found at {engine_path}. Advanced channel will fail if invoked.")
    else:
        print("[WARNING] vllm package missing. Advanced channel unavailable.")


# ==========================================
# CHANNEL A: PRIVATE TENSORRT-LLM INFERENCE
# ==========================================
async def stream_tensorrt_engine(prompt: str, max_tokens: int, temperature: float) -> AsyncGenerator[str, None]:
    """Streams data dynamically from the local multi-GPU compiled architecture."""
    request_id = random_uuid()
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.95
    )
    
    results_generator = tensorrt_engine_instance.generate(prompt, sampling_params, request_id)
    
    async for request_output in results_generator:
        if request_output.outputs:
            # Yield raw delta tokens directly to the streaming connection
            yield request_output.outputs[-1].text
        await asyncio.sleep(0)  # Yield execution control back to the event loop


# ==========================================
# CHANNEL B: GROQ CLOUD CLOUD INFERENCE
# ==========================================
async def stream_groq_api(prompt: str, max_tokens: int, temperature: float) -> AsyncGenerator[str, None]:
    """Streams data instantly from the cloud hardware using the Llama brain configuration."""
    try:
        completion = await groq_client.chat.completions.create(
            model="llama3-70b-8192",  # Replace with your targeted Llama core variant on Groq
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in completion:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        yield f"\n[Groq Channel Error]: {str(e)}"


# ==========================================
# CORE HYBRID DUAL-MODE ROUTER
# ==========================================
@app.post("/v1/supreme/chat")
async def process_hybrid_route(request: SupremeRequest):
    """Dynamically channels incoming workflow streams based on specific operational constraints."""
    
    # --- ROUTE 1: ADVANCED CHANNELD INFERENCE (TensorRT-LLM / Local Secure Core) ---
    if request.mode == "advanced":
        if not tensorrt_engine_instance:
            raise HTTPException(
                status_code=503, 
                detail="Advanced TensorRT Core engine is not loaded or configured on this serverless instance."
            )
        
        # Inject private system instructions/reasoning structural markers
        structured_prompt = f"<system>Initialize advanced structural cognitive mode.</system>\n{request.prompt}"
        
        return StreamingResponse(
            stream_tensorrt_engine(structured_prompt, request.max_tokens, request.temperature),
            media_type="text/event-stream"
        )
        
    # --- ROUTE 2: STANDARD PERFORMANCE INFERENCE (Groq Cloud Speed Core) ---
    else:
        if not groq_client:
            # Fallback check if Groq credentials fail but local engine is active
            if tensorrt_engine_instance:
                print("[FALLBACK] Groq unavailable. Dynamically routing to local core.")
                return StreamingResponse(
                    stream_tensorrt_engine(request.prompt, request.max_tokens, request.temperature),
                    media_type="text/event-stream"
                )
            raise HTTPException(
                status_code=503, 
                detail="Groq API client is uninitialized. Verify your GROQ_API_KEY environment variables."
            )
            
        return StreamingResponse(
            stream_groq_api(request.prompt, request.max_tokens, request.temperature),
            media_type="text/event-stream"
        )

# Health check validation path for serverless load balancers
@app.get("/health")
async def health_check():
    return {
        "status": "active",
        "groq_channel": groq_client is not None,
        "tensorrt_channel": tensorrt_engine_instance is not None
    }
