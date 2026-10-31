import os
import json
import asyncio
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

app = FastAPI(title="Supreme AI Hybrid Routing Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SupremeRequest(BaseModel):
    prompt: str
    mode: str = "standard"  
    max_tokens: int = 1024
    temperature: float = 0.7

tensorrt_engine_instance = None
groq_client = None
engine_loading_status = "uninitialized"  # Tracks background loading state

# Helper function to load the heavy vLLM engine in the background
def load_vllm_engine_background():
    global tensorrt_engine_instance, engine_loading_status
    engine_path = "./supreme-compiled-engine"
    if os.path.exists(engine_path):
        try:
            print(f"[INFO] Asynchronously compiling and loading TensorRT engine from {engine_path}...")
            engine_loading_status = "loading"
            engine_args = EngineArgs(
                model=engine_path,
                tensor_parallel_size=2,       
                block_size=16,                
                gpu_memory_utilization=0.90,  
                enforce_eager=True            
            )
            tensorrt_engine_instance = LLMEngine.from_engine_args(engine_args)
            engine_loading_status = "ready"
            print("[INFO] Advanced TensorRT-LLM Core successfully initialized.")
        except Exception as e:
            engine_loading_status = f"failed: {str(e)}"
            print(f"[ERROR] Failed to load TensorRT Engine: {str(e)}")
    else:
        engine_loading_status = "missing_directory"
        print(f"[WARNING] TensorRT engine directory not found at {engine_path}.")

@app.on_event("startup")
async def initialize_system_channels():
    global groq_client
    
    # 1. Initialize Groq Channel immediately (Non-blocking)
    if GROQ_AVAILABLE:
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            groq_client = AsyncGroq(api_key=api_key)
            print("[INFO] Groq Channel successfully initialized.")
        else:
            print("[WARNING] GROQ_API_KEY not found in environment variables.")
    else:
        print("[WARNING] groq package missing.")

    # 2. Kick off the heavy vLLM engine loading as a background task
    if VLLM_AVAILABLE:
        # loop.run_in_executor offloads blocking code so FastAPI can open its port instantly!
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, load_vllm_engine_background)
    else:
        print("[WARNING] vllm package missing. Advanced channel unavailable.")

async def stream_tensorrt_engine(prompt: str, max_tokens: int, temperature: float) -> AsyncGenerator[str, None]:
    if not VLLM_AVAILABLE or not tensorrt_engine_instance:
        yield f"[System Error]: TensorRT core engine uninitialized. Status: {engine_loading_status}"
        return
        
    request_id = random_uuid()
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.95
    )
    
    results_generator = tensorrt_engine_instance.generate(prompt, sampling_params, request_id)
    
    async for request_output in results_generator:
        if request_output.outputs:
            yield request_output.outputs[-1].text
        await asyncio.sleep(0)

async def stream_groq_api(prompt: str, max_tokens: int, temperature: float) -> AsyncGenerator[str, None]:
    try:
        completion = await groq_client.chat.completions.create(
            model="llama3-70b-8192",  
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in completion:
            delta = chunk.choices.delta.content
            if delta:
                yield delta
    except Exception as e:
        yield f"\n[Groq Channel Error]: {str(e)}"

@app.post("/v1/supreme/chat")
async def process_hybrid_route(request: SupremeRequest):
    if request.mode == "advanced":
        if engine_loading_status == "loading":
            raise HTTPException(status_code=503, detail="TensorRT engine is still booting up in the background. Please retry in a moment.")
        if not tensorrt_engine_instance:
            raise HTTPException(status_code=503, detail=f"Advanced engine unavailable. Status: {engine_loading_status}")
            
        structured_prompt = f"<system>Initialize advanced structural cognitive mode.</system>\n{request.prompt}"
        return StreamingResponse(
            stream_tensorrt_engine(structured_prompt, request.max_tokens, request.temperature),
            media_type="text/event-stream"
        )
    else:
        if not groq_client:
            if tensorrt_engine_instance:
                print("[FALLBACK] Groq unavailable. Dynamically routing to local core.")
                return StreamingResponse(
                    stream_tensorrt_engine(request.prompt, request.max_tokens, request.temperature),
                    media_type="text/event-stream"
                )
            raise HTTPException(
                status_code=503, 
                detail=f"Groq API client uninitialized and local core state is: {engine_loading_status}"
            )
        return StreamingResponse(
            stream_groq_api(request.prompt, request.max_tokens, request.temperature),
            media_type="text/event-stream"
        )

@app.get("/health")
async def health_check():
    return {
        "status": "active",
        "groq_channel": groq_client is not None,
        "tensorrt_channel_status": engine_loading_status
    }

if __name__ == "__main__":
    import uvicorn
    server_port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=server_port, log_level="info")
