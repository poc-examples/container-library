import os
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoTokenizer, AutoModel

# Define the model path for the embeddings model
model_path = "jinaai/jina-embeddings-v2-base-en"
precision = torch.float16  # Define the precision for the model

# Initialize FastAPI app
app = FastAPI()

# Define a request model for incoming data
class TextRequest(BaseModel):
    text: str
    # texts: List[str]  # A list of texts to encode

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, torch_dtype=precision)

# Check if CUDA is available and set the model to use GPU if available
if torch.cuda.is_available():
    model = model.to('cuda')

# Health check endpoint
@app.get("/.well-known/live")
async def live_check():
    return Response(status_code=204)

# Readiness check endpoint
@app.get("/.well-known/ready")
async def ready_check():
    return Response(status_code=204)

# Meta information endpoint
@app.get("/meta")
async def meta_info():
    return {
        "name": "CustomVectorizer",
        "description": "A vectorizer endpoint for text2vec.",
        "type": "text2vec",
        "model": model_path,
        "precision": "float16",
        "cuda": "true"
    }

# Define the endpoint for obtaining embeddings
# https://weaviate.io/developers/weaviate/modules/other-modules/custom-modules
@app.post("/vectors")
@app.post("/vectors/")
async def vectors(request: TextRequest):
    print(request)
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="Input text list is empty")
    
    # Encode texts
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=5000)
    
    # Move inputs to GPU if available
    if torch.cuda.is_available():
        inputs = {name: tensor.to('cuda') for name, tensor in inputs.items()}
    
    # Obtain embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    
    # Capture the Shape and respond for weaviate requirements
    dim = embeddings.shape[-1]

    # Move embeddings back to CPU and convert to list
    embeddings = embeddings.cpu().numpy().tolist()
    
    return {
        "text": text,
        "vector": embeddings[0],
        "dim": dim
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
