# main.py
import gensim.models.fasttext
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# --- Configuration ---
MODEL_PATH = "../../Model_Training/Models/cc.hi.300.bin" # Make sure this file is accessible

# --- Load Model (once on startup) ---
try:
    model = gensim.models.fasttext.load_facebook_model(MODEL_PATH)
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Please download cc.hi.300.bin or update MODEL_PATH.")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- FastAPI App ---
app = FastAPI()

# --- Pydantic Models for Request/Response ---
class InputData(BaseModel):
    texts: List[str]

class OutputData(BaseModel):
    embeddings: List[List[float]]

# --- API Endpoint ---
@app.post("/embeddings", response_model=OutputData)
async def get_embeddings(data: InputData):
    """
    Receives a list of strings and returns their FastText embeddings.
    """
    embeddings = [model.wv[text].tolist() for text in data.texts]
    return {"embeddings": embeddings}

# --- To run the app (optional, usually done via command line) ---
if __name__ == "__main__":
    import uvicorn
    print(f"Starting server. Access API at http://127.0.0.1:8000")
    print(f"API Docs available at http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)
    # uvicorn main:app --reload