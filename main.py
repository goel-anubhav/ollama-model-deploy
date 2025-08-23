# main.py
import asyncio
import aiofiles
import aiohttp
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import logging
import time
import json
from datetime import datetime
from typing import Optional, AsyncGenerator
import signal
import sys
from contextlib import asynccontextmanager
import multiprocessing
import uvloop
import orjson

# Use uvloop for better async performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# === Enhanced Config ===
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "ollama")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", 11434))
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
OLLAMA_API_GENERATE = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_API_MODELS = f"{OLLAMA_BASE_URL}/api/tags"
OLLAMA_API_PULL = f"{OLLAMA_BASE_URL}/api/pull"
MODEL_NAME = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")
CHAT_LOG_FILE = "chat_history.txt"

# Performance and resource optimization settings
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", multiprocessing.cpu_count() * 4))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 8192))
CONNECTION_POOL_SIZE = int(os.getenv("CONNECTION_POOL_SIZE", 100))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 0))  # 0 means no timeout
KEEP_ALIVE_TIMEOUT = int(os.getenv("KEEP_ALIVE_TIMEOUT", 300))

# Global connection session
session: Optional[aiohttp.ClientSession] = None
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# === Enhanced Models ===
class Question(BaseModel):
    question: str
    stream: Optional[bool] = False
    include_context: Optional[bool] = False
    context_limit: Optional[int] = 5
    max_tokens: Optional[int] = None  # No limit by default
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 40

# === Async Session Management ===
async def get_session() -> aiohttp.ClientSession:
    """Get or create HTTP session with optimized settings"""
    global session
    if session is None or session.closed:
        # Create connector with optimized settings
        connector = aiohttp.TCPConnector(
            limit=CONNECTION_POOL_SIZE,
            limit_per_host=CONNECTION_POOL_SIZE,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=KEEP_ALIVE_TIMEOUT,
            enable_cleanup_closed=True,
            force_close=False
        )
        
        # Create timeout with no limits
        timeout = aiohttp.ClientTimeout(
            total=REQUEST_TIMEOUT if REQUEST_TIMEOUT > 0 else None,
            connect=30,
            sock_read=None,
            sock_connect=10
        )
        
        session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            json_serialize=lambda obj: orjson.dumps(obj).decode()
        )
        logger.info("Created new HTTP session with optimized settings")
    
    return session

async def close_session():
    """Cleanup HTTP session"""
    global session
    if session and not session.closed:
        await session.close()
        logger.info("HTTP session closed")

# === Async File Operations ===
async def save_chat_to_file_async(question: str, answer: str):
    """Asynchronously save chat interaction to a text file"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        chat_content = f"=== Chat Entry - {timestamp} ===\n"
        chat_content += f"Question: {question}\n"
        chat_content += f"Answer: {answer}\n"
        chat_content += "-" * 50 + "\n\n"
        
        async with aiofiles.open(CHAT_LOG_FILE, "a", encoding="utf-8") as f:
            await f.write(chat_content)
        
        logger.info(f"Chat saved to {CHAT_LOG_FILE}")
    except Exception as e:
        logger.error(f"Failed to save chat: {e}")

async def get_recent_chat_history_async(limit: int = 5) -> str:
    """Asynchronously get recent chat history for context"""
    try:
        if not os.path.exists(CHAT_LOG_FILE):
            return ""
        
        async with aiofiles.open(CHAT_LOG_FILE, "r", encoding="utf-8") as f:
            content = await f.read()
        
        # Parse chat entries efficiently
        entries = []
        current_entry = {}
        lines = content.split('\n')
        
        for line in lines:
            if line.startswith("=== Chat Entry -"):
                if current_entry:
                    entries.append(current_entry)
                current_entry = {"timestamp": line.split(" - ")[1].split(" ===")[0]}
            elif line.startswith("Question: "):
                current_entry["question"] = line[10:]
            elif line.startswith("Answer: "):
                current_entry["answer"] = line[8:]
        
        if current_entry and "question" in current_entry:
            entries.append(current_entry)
        
        # Get recent entries
        recent_entries = entries[-limit:] if entries else []
        
        # Build context efficiently
        if not recent_entries:
            return ""
        
        context_parts = ["Previous conversation context:\n"]
        for i, entry in enumerate(recent_entries, 1):
            context_parts.extend([
                f"\nConversation {i}:\n",
                f"User: {entry.get('question', 'N/A')}\n",
                f"Assistant: {entry.get('answer', 'N/A')}\n"
            ])
        
        context_parts.extend([
            "\n" + "="*50 + "\n",
            "Based on the above conversation history, please respond to the following new question:\n\n"
        ])
        
        return "".join(context_parts)
        
    except Exception as e:
        logger.error(f"Failed to get chat history: {e}")
        return ""

# === Enhanced Model Management ===
async def ensure_model_installed_async():
    """Asynchronously check and install model"""
    session = await get_session()
    
    try:
        logger.info(f"Checking if model '{MODEL_NAME}' is installed...")
        
        async with session.get(OLLAMA_API_MODELS) as response:
            response.raise_for_status()
            data = await response.json()
            models = data.get("models", [])
            model_names = [m.get("name") for m in models]

            if MODEL_NAME not in model_names:
                logger.info(f"⬇️ Model '{MODEL_NAME}' not found. Pulling...")
                
                async with session.post(OLLAMA_API_PULL, json={"name": MODEL_NAME}) as pull_response:
                    pull_response.raise_for_status()
                    logger.info(f"✅ Model '{MODEL_NAME}' downloaded successfully.")
            else:
                logger.info(f"✅ Model '{MODEL_NAME}' already installed.")

    except Exception as e:
        logger.error(f"❌ Failed to check/download model: {e}")
        raise

async def preload_model_async():
    """Asynchronously preload model into VRAM"""
    session = await get_session()
    
    try:
        logger.info(f"Preloading model '{MODEL_NAME}' into VRAM...")
        
        payload = {
            "model": MODEL_NAME,
            "prompt": "Hello",
            "stream": False,
            "options": {
                "num_predict": 1,  # Minimal prediction for preload
                "temperature": 0.1
            }
        }
        
        async with session.post(OLLAMA_API_GENERATE, json=payload) as response:
            response.raise_for_status()
            logger.info(f"✅ Model '{MODEL_NAME}' preloaded successfully.")
            
    except Exception as e:
        logger.error(f"⚠️ Model preload failed: {e}")

# === Async Streaming Generator ===
async def generate_stream_response_async(
    original_question: str, 
    contextual_prompt: str, 
    question_params: Question
) -> AsyncGenerator[str, None]:
    """Enhanced async generator for streaming responses"""
    session = await get_session()
    
    async with request_semaphore:  # Control concurrent requests
        try:
            # Build comprehensive payload
            payload = {
                "model": MODEL_NAME,
                "prompt": contextual_prompt,
                "stream": True,
                "options": {
                    "temperature": question_params.temperature,
                    "top_p": question_params.top_p,
                    "top_k": question_params.top_k,
                    "repeat_penalty": 1.1,
                    "num_ctx": 32768,  # Large context window
                }
            }
            
            # Remove token limits - let model generate freely
            if question_params.max_tokens:
                payload["options"]["num_predict"] = question_params.max_tokens
            
            logger.info(f"Starting streaming request - Context: {len(contextual_prompt)} chars")
            
            full_answer = ""
            
            async with session.post(OLLAMA_API_GENERATE, json=payload) as response:
                response.raise_for_status()
                
                async for line in response.content:
                    if line:
                        try:
                            line_str = line.decode('utf-8').strip()
                            if not line_str:
                                continue
                                
                            chunk = orjson.loads(line_str)
                            
                            if 'response' in chunk:
                                response_text = chunk['response']
                                full_answer += response_text
                                
                                # Yield chunk with better formatting
                                yield f"data: {orjson.dumps({'response': response_text}).decode()}\n\n"
                                
                            if chunk.get('done', False):
                                # Save complete conversation
                                await save_chat_to_file_async(original_question, full_answer.strip())
                                yield f"data: {orjson.dumps({'done': True, 'total_tokens': len(full_answer)}).decode()}\n\n"
                                break
                                
                        except (orjson.JSONDecodeError, UnicodeDecodeError) as e:
                            logger.warning(f"Parse error (continuing): {e}")
                            continue
                            
        except Exception as e:
            logger.error(f"Streaming request failed: {e}")
            yield f"data: {orjson.dumps({'error': str(e)}).decode()}\n\n"

# === Enhanced Non-Streaming Response ===
async def get_ollama_response_async(contextual_prompt: str, question_params: Question) -> str:
    """Enhanced async non-streaming response"""
    session = await get_session()
    
    async with request_semaphore:
        try:
            payload = {
                "model": MODEL_NAME,
                "prompt": contextual_prompt,
                "stream": False,
                "options": {
                    "temperature": question_params.temperature,
                    "top_p": question_params.top_p,
                    "top_k": question_params.top_k,
                    "repeat_penalty": 1.1,
                    "num_ctx": 32768,  # Large context window
                }
            }
            
            # Remove token limits
            if question_params.max_tokens:
                payload["options"]["num_predict"] = question_params.max_tokens
            
            logger.info(f"Sending request - Context: {len(contextual_prompt)} chars")
            
            async with session.post(OLLAMA_API_GENERATE, json=payload) as response:
                response.raise_for_status()
                data = await response.json()

                if "response" not in data:
                    raise HTTPException(status_code=500, detail="Ollama returned no 'response' field")

                return data["response"].strip()
                
        except Exception as e:
            logger.error(f"Request to Ollama failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# === Application Lifespan Management ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown"""
    # Startup
    logger.info("Starting application...")
    
    # Initialize chat log file
    if not os.path.exists(CHAT_LOG_FILE):
        async with aiofiles.open(CHAT_LOG_FILE, "w", encoding="utf-8") as f:
            await f.write(f"=== Chat History Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    
    # Wait for Ollama and setup
    await wait_for_ollama()
    await ensure_model_installed_async()
    await preload_model_async()
    
    logger.info("✅ Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    await close_session()
    logger.info("✅ Application shutdown complete")

async def wait_for_ollama():
    """Wait for Ollama server to be ready"""
    session = await get_session()
    
    for attempt in range(60):  # Increased attempts
        try:
            async with session.get(OLLAMA_BASE_URL) as response:
                if response.status == 200:
                    logger.info("✅ Ollama server is ready")
                    return
        except Exception:
            logger.info(f"⏳ Waiting for Ollama... (attempt {attempt + 1}/60)")
        
        await asyncio.sleep(3)
    
    logger.error("❌ Ollama server did not start in time")
    raise RuntimeError("Ollama server unavailable")

# === FastAPI App with Lifespan ===
app = FastAPI(
    title="Enhanced Ollama FastAPI Gateway",
    description="High-performance async Ollama gateway with unlimited tokens",
    version="2.0.0",
    lifespan=lifespan
)

# === API Endpoints ===
@app.post("/ask")
async def ask_ollama(q: Question, background_tasks: BackgroundTasks):
    """Enhanced ask endpoint with full async support"""
    try:
        # Build contextual prompt
        contextual_prompt = q.question
        if q.include_context:
            context = await get_recent_chat_history_async(q.context_limit)
            if context:
                contextual_prompt = f"{context}{q.question}"
        
        if q.stream:
            # Return streaming response
            return StreamingResponse(
                generate_stream_response_async(q.question, contextual_prompt, q),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            # Non-streaming response
            answer = await get_ollama_response_async(contextual_prompt, q)
            
            # Save chat in background
            background_tasks.add_task(save_chat_to_file_async, q.question, answer)
            
            return {
                "answer": answer,
                "context_used": q.include_context,
                "context_limit": q.context_limit if q.include_context else 0,
                "tokens": len(answer.split()),
                "characters": len(answer)
            }
            
    except Exception as e:
        logger.error(f"Error in ask_ollama: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat-history")
async def get_chat_history():
    """Get raw chat history"""
    try:
        if not os.path.exists(CHAT_LOG_FILE):
            return {"message": "No chat history found"}
        
        async with aiofiles.open(CHAT_LOG_FILE, "r", encoding="utf-8") as f:
            content = await f.read()
        
        return {"chat_history": content}
    except Exception as e:
        logger.error(f"Failed to read chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat-history/formatted")
async def get_formatted_chat_history(limit: Optional[int] = 10):
    """Get formatted chat history"""
    try:
        context = await get_recent_chat_history_async(limit)
        if not context:
            return {"message": "No chat history found", "formatted_history": ""}
        
        return {
            "message": f"Last {limit} conversations",
            "formatted_history": context
        }
    except Exception as e:
        logger.error(f"Failed to get formatted chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat-history")
async def clear_chat_history():
    """Clear chat history"""
    try:
        async with aiofiles.open(CHAT_LOG_FILE, "w", encoding="utf-8") as f:
            await f.write(f"=== Chat History Log - Cleared at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
        
        return {"message": "Chat history cleared successfully"}
    except Exception as e:
        logger.error(f"Failed to clear chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check with system info"""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        "session_active": session is not None and not session.closed,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "model": MODEL_NAME,
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        "connection_pool_size": CONNECTION_POOL_SIZE,
        "chunk_size": CHUNK_SIZE,
        "request_timeout": REQUEST_TIMEOUT,
        "chat_log_exists": os.path.exists(CHAT_LOG_FILE),
        "chat_log_size": os.path.getsize(CHAT_LOG_FILE) if os.path.exists(CHAT_LOG_FILE) else 0
    }

# === Graceful Shutdown Handler ===
async def shutdown_handler():
    """Handle graceful shutdown"""
    logger.info("Received shutdown signal")
    await close_session()

# Register signal handlers
if __name__ == "__main__":
    import uvicorn
    
    # Configure uvicorn for high performance
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        loop="uvloop",
        http="httptools",
        workers=1,  # Use 1 worker with async
        log_level="info",
        access_log=True,
        use_colors=True,
        server_header=False,
        date_header=False
    )