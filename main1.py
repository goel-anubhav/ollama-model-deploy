# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
import logging
import time

logging.basicConfig(level=logging.INFO)

# === Config ===
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "ollama")  # Docker service/container name for Ollama
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", 11434))
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
OLLAMA_API_GENERATE = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_API_MODELS = f"{OLLAMA_BASE_URL}/api/tags"
OLLAMA_API_PULL = f"{OLLAMA_BASE_URL}/api/pull"
MODEL_NAME = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

# === FastAPI App ===
app = FastAPI(title="Ollama FastAPI Gateway")

class Question(BaseModel):
    question: str

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

# === API Endpoint ===
@app.post("/ask")
def ask_ollama(q: Question):
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

        return {"answer": data["response"].strip()}

    except requests.RequestException as e:
        logging.error(f"Request to Ollama failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
