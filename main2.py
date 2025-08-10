# main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
import os
import logging
import time
import json
from datetime import datetime
from typing import Optional

logging.basicConfig(level=logging.INFO)

# === Config ===
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "ollama")  # Docker service/container name for Ollama
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", 11434))
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
OLLAMA_API_GENERATE = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_API_MODELS = f"{OLLAMA_BASE_URL}/api/tags"
OLLAMA_API_PULL = f"{OLLAMA_BASE_URL}/api/pull"
MODEL_NAME = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
CHAT_LOG_FILE = "chat_history.txt"

# === FastAPI App ===
app = FastAPI(title="Ollama FastAPI Gateway")

class Question(BaseModel):
    question: str
    stream: Optional[bool] = False

# === Chat Storage Functions ===
def save_chat_to_file(question: str, answer: str):
    """Save chat interaction to a text file"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chat_entry = {
            "timestamp": timestamp,
            "question": question,
            "answer": answer
        }
        
        with open(CHAT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"=== Chat Entry - {timestamp} ===\n")
            f.write(f"Question: {question}\n")
            f.write(f"Answer: {answer}\n")
            f.write("-" * 50 + "\n\n")
        
        logging.info(f"Chat saved to {CHAT_LOG_FILE}")
    except Exception as e:
        logging.error(f"Failed to save chat: {e}")

def save_streaming_chat_to_file(question: str, full_answer: str):
    """Save streaming chat interaction to a text file"""
    save_chat_to_file(question, full_answer)

# === Check & Download Model via HTTP API ===
def ensure_model_installed():
    try:
        logging.info(f"Checking if model '{MODEL_NAME}' is installed via Ollama API...")
        r = requests.get(OLLAMA_API_MODELS, timeout=30)
        r.raise_for_status()
        models = r.json().get("models", [])
        model_names = [m.get("name") for m in models]

        if MODEL_NAME not in model_names:
            logging.info(f"⬇️ Model '{MODEL_NAME}' not found. Pulling from Ollama...")
            pull_req = requests.post(OLLAMA_API_PULL, json={"name": MODEL_NAME}, timeout=600)
            pull_req.raise_for_status()
            logging.info(f"✅ Model '{MODEL_NAME}' downloaded successfully.")
        else:
            logging.info(f"✅ Model '{MODEL_NAME}' already installed.")

    except requests.RequestException as e:
        logging.error(f"❌ Failed to check/download model: {e}")

# === Preload Model into VRAM ===
def preload_model():
    try:
        logging.info(f"Preloading model '{MODEL_NAME}' into VRAM...")
        payload = {
            "model": MODEL_NAME,
            "prompt": "Hello",
            "stream": False
        }
        response = requests.post(OLLAMA_API_GENERATE, json=payload, timeout=600)
        response.raise_for_status()
        logging.info(f"✅ Model '{MODEL_NAME}' preloaded successfully.")
    except requests.RequestException as e:
        logging.error(f"⚠️ Model preload failed: {e}")

# === Startup Event ===
@app.on_event("startup")
def startup_event():
    # Initialize chat log file
    if not os.path.exists(CHAT_LOG_FILE):
        with open(CHAT_LOG_FILE, "w", encoding="utf-8") as f:
            f.write(f"=== Chat History Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    
    # Wait for Ollama server to be ready
    for _ in range(40):
        try:
            r = requests.get(OLLAMA_BASE_URL, timeout=5)
            if r.status_code == 200:
                logging.info("✅ Ollama server is up.")
                break
        except:
            logging.info("⏳ Waiting for Ollama to be ready...")
        time.sleep(3)
    else:
        logging.error("❌ Ollama server did not start in time.")
        return

    ensure_model_installed()
    preload_model()

# === Streaming Generator ===
def generate_stream_response(question: str):
    """Generator function for streaming responses"""
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": question,
            "stream": True
        }
        
        logging.info(f"Sending streaming request to Ollama: {question}")
        response = requests.post(OLLAMA_API_GENERATE, json=payload, stream=True, timeout=300)
        response.raise_for_status()
        
        full_answer = ""
        
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode('utf-8'))
                    if 'response' in chunk:
                        response_text = chunk['response']
                        full_answer += response_text
                        
                        # Yield the chunk as JSON
                        yield f"data: {json.dumps({'response': response_text})}\n\n"
                        
                    if chunk.get('done', False):
                        # Save the complete conversation when streaming is done
                        save_streaming_chat_to_file(question, full_answer.strip())
                        yield f"data: {json.dumps({'done': True})}\n\n"
                        break
                        
                except json.JSONDecodeError as e:
                    logging.error(f"JSON decode error: {e}")
                    continue
                    
    except requests.RequestException as e:
        logging.error(f"Streaming request to Ollama failed: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

# === API Endpoints ===
@app.post("/ask")
def ask_ollama(q: Question):
    if q.stream:
        # Return streaming response
        return StreamingResponse(
            generate_stream_response(q.question),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
    else:
        # Non-streaming response
        try:
            payload = {
                "model": MODEL_NAME,
                "prompt": q.question,
                "stream": False
            }
            logging.info(f"Sending to Ollama: {payload}")
            response = requests.post(OLLAMA_API_GENERATE, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()

            if "response" not in data:
                raise HTTPException(status_code=500, detail="Ollama returned no 'response' field.")

            answer = data["response"].strip()
            
            # Save the chat to file
            save_chat_to_file(q.question, answer)
            
            return {"answer": answer}

        except requests.RequestException as e:
            logging.error(f"Request to Ollama failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# === Chat History Endpoint ===
@app.get("/chat-history")
def get_chat_history():
    """Endpoint to retrieve chat history"""
    try:
        if not os.path.exists(CHAT_LOG_FILE):
            return {"message": "No chat history found"}
        
        with open(CHAT_LOG_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        
        return {"chat_history": content}
    except Exception as e:
        logging.error(f"Failed to read chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === Clear Chat History Endpoint ===
@app.delete("/chat-history")
def clear_chat_history():
    """Endpoint to clear chat history"""
    try:
        with open(CHAT_LOG_FILE, "w", encoding="utf-8") as f:
            f.write(f"=== Chat History Log - Cleared at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
        
        return {"message": "Chat history cleared successfully"}
    except Exception as e:
        logging.error(f"Failed to clear chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === Health Check ===
@app.get("/health")
def health_check():
    return {"status": "healthy", "model": MODEL_NAME}
