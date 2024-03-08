from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama
import subprocess
import logging
from huggingface_hub import hf_hub_download
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextRequest(BaseModel):
    input_text: str

model_path = "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF"
filename = "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
quantization = "models--TheBloke--Mixtral-8x7B-Instruct-v0.1-GGUF/snapshots/fa1d3835c5d45a3a74c0b68805fcdc133dba2b6a/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
save_directory = "/ai-models/mistralai"

logger.info(f"Model Path: {model_path}")
logger.info(f"Download File: {filename}")
logger.info(f"Quantization: {quantization}")
logger.info(f"Model Location: {save_directory}")

logger.info("""
We are checking to see if the model is available locally.
If Model is not available it will be downloaded.
File sizes are large and may take some time to complete.
""")

hf_hub_download(
    repo_id=model_path, 
    filename=filename,
    cache_dir=save_directory
)

logger.info(f"{subprocess.check_output(['ls', '-la', f'{save_directory}/{quantization}' ], text=True)}\n")

logger.info("Starting Model...")
try:
    logger.info(f"Initializing Llama with model path: {save_directory}/{quantization}")
    llm = Llama(
        model_path=f"{save_directory}/{quantization}",
        n_ctx=28000,  # The max sequence length to use - note that longer sequence lengths require much more resources
        n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
        n_gpu_layers=35
    )
except Exception as e:
    logger.error(e)

logger.info(f"Llama Loaded...")

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/generate-text")
async def generate_text(request: TextRequest):
    prompt = request.input_text

    output = llm(
      f"[INST] {prompt} [/INST]", # Prompt
      max_tokens=28000,  # Generate up to 512 tokens
      stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
      echo=False        # Whether to echo the prompt
    )

    # Extract the relevant information from the output
    response_text = output.get("choices")[0]["text"]

    return {"generated_text": f"{response_text}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
