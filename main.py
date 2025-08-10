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
    include_context: Optional[bool] = False
    context_limit: Optional[int] = 5  # Number of previous conversations to include

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

def get_recent_chat_history(limit: int = 5):
    """Get recent chat history for context"""
    try:
        if not os.path.exists(CHAT_LOG_FILE):
            return ""
        
        with open(CHAT_LOG_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Parse chat entries
        entries = []
        current_entry = {}
        lines = content.split('\n')
        
        for line in lines:
            if line.startswith("=== Chat Entry -"):
                if current_entry:
                    entries.append(current_entry)
                current_entry = {"timestamp": line.split(" - ")[1].split(" ===")[0]}
            elif line.startswith("Question: "):
                current_entry["question"] = line[10:]  # Remove "Question: "
            elif line.startswith("Answer: "):
                current_entry["answer"] = line[8:]  # Remove "Answer: "
        
        # Add the last entry if it exists
        if current_entry and "question" in current_entry:
            entries.append(current_entry)
        
        # Get the most recent entries (excluding current question)
        recent_entries = entries[-limit:] if len(entries) > 0 else []
        
        # Format context
        context = ""
        if recent_entries:
            context = "Previous conversation context:\n"
            for i, entry in enumerate(recent_entries, 1):
                context += f"\nConversation {i}:\n"
                context += f"User: {entry.get('question', 'N/A')}\n"
                context += f"Assistant: {entry.get('answer', 'N/A')}\n"
            
            context += "\n" + "="*50 + "\n"
            context += "Based on the above conversation history, please respond to the following new question:\n\n"
        
        return context
        
    except Exception as e:
        logging.error(f"Failed to get chat history: {e}")
        return ""

def build_contextual_prompt(question: str, include_context: bool = False, context_limit: int = 5):
    """Build prompt with or without chat history context"""
    if not include_context:
        return question
    
    context = get_recent_chat_history(context_limit)
    if context:
        return f"{context}{question}"
    else:
        return question

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
def generate_stream_response(original_question: str, contextual_prompt: str):
    """Generator function for streaming responses"""
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": contextual_prompt,
            "stream": True
        }
        
        logging.info(f"Sending streaming request to Ollama with context: {len(contextual_prompt)} characters")
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
                        # Save only the original question, not the contextual prompt
                        save_streaming_chat_to_file(original_question, full_answer.strip())
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
    # Build contextual prompt
    contextual_prompt = build_contextual_prompt(
        q.question, 
        q.include_context, 
        q.context_limit
    )
    
    if q.stream:
        # Return streaming response
        return StreamingResponse(
            generate_stream_response(q.question, contextual_prompt),
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
                "prompt": contextual_prompt,
                "stream": False
            }
            logging.info(f"Sending to Ollama with context: {len(contextual_prompt)} characters")
            response = requests.post(OLLAMA_API_GENERATE, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()

            if "response" not in data:
                raise HTTPException(status_code=500, detail="Ollama returned no 'response' field.")

            answer = data["response"].strip()
            
            # Save the chat to file (save original question, not contextual prompt)
            save_chat_to_file(q.question, answer)
            
            return {
                "answer": answer,
                "context_used": q.include_context,
                "context_limit": q.context_limit if q.include_context else 0
            }

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

# === Get Formatted Chat History ===
@app.get("/chat-history/formatted")
def get_formatted_chat_history(limit: Optional[int] = 10):
    """Endpoint to retrieve formatted chat history"""
    try:
        context = get_recent_chat_history(limit)
        if not context:
            return {"message": "No chat history found", "formatted_history": ""}
        
        return {
            "message": f"Last {limit} conversations",
            "formatted_history": context
        }
    except Exception as e:
        logging.error(f"Failed to get formatted chat history: {e}")
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
