import os
import gdown
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI()

MODEL_PATH = "model.gguf"
# Aapka Google Drive ka direct link yahan set kar diya gaya hai
DRIVE_URL = "https://drive.google.com/file/d/15nb88yCRzvBb7DCf-AZ3I28Jw7_NX97j/view?usp=drivesdk"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        gdown.download(DRIVE_URL, MODEL_PATH, quiet=False, fuzzy=True)
        print("Download complete!")

# Server start hone par model download aur load hoga
download_model()
llm = Llama(model_path=MODEL_PATH, n_ctx=512, n_threads=2)

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 128

@app.post("/generate")
def generate_text(req: PromptRequest):
    try:
        output = llm(req.prompt, max_tokens=req.max_tokens, stop=["\n"], echo=False)
        return {"response": output["choices"][0]["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"status": "Model API is running successfully!"}
