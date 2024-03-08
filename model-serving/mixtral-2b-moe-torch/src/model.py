import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import torch
from torch.autograd import profiler
import time
import subprocess
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

model_path = "cloudyu/Mixtral_7Bx2_MoE"
save_directory = "/a-models/mistralai"

# float32
# float16
# int8
precision = torch.float16

print(f"loading model: {model_path}")
print(f"model location: {save_directory}")
print(f"precision: {precision}\n")

print(f"Available CPU Cores: {subprocess.check_output(['nproc'], text=True)}")
print(f"{subprocess.check_output(['nvcc', '--version'], text=True)}\n")

print(f"CUDA AVAILABLE: {torch.cuda.is_available()}")

os.environ["HF_HOME"] = save_directory
from transformers import AutoTokenizer, AutoModelForCausalLM

# huggingface_hub.snapshot_download(repo_id=model_path, repo_type="model", max_workers=1)

# Function to list all GPUs
def list_gpus():
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs available.")
    else:
        print(f"Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Call the function to list GPUs
list_gpus()

class TextRequest(BaseModel):
    input_text: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

if tokenizer.eos_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
else:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=precision,
    bnb_4bit_compute_dtype=precision, 
    device_map='auto', 
    local_files_only=False, 
    load_in_4bit=True,
    use_safetensors=True,
    # low_cpu_mem_usage=True
)


# Check if multiple GPUs are available and wrap the model
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

@app.post("/generate-text")
async def generate_text(request: TextRequest):
    input_text = request.input_text
    if not input_text:
        raise HTTPException(status_code=400, detail="Input text is empty")

    encoding = tokenizer(
        input_text, 
        return_tensors="pt",
        padding="max_length",  # Pad to max_length
        truncation=True,       # Truncate to max_length
        max_length=512         # Specify the max_length
    )
    input_ids = encoding.input_ids

    attention_mask = encoding["attention_mask"]

    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")

    start_time = time.time()
    # with profiler.profile(profile_memory=True, record_shapes=True, use_cuda=torch.cuda.is_available()) as prof:
    generation_output = model.module.generate(
        input_ids, 
        attention_mask=attention_mask,
        max_new_tokens=2500, 
        repetition_penalty=1.2)
    end_time = time.time()

    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    generated_text = tokenizer.decode(generation_output[0], skip_special_tokens=True)
    
    # Calculate tokens per second
    num_tokens = len(tokenizer.tokenize(generated_text))
    inference_time = end_time - start_time
    tokens_per_second = num_tokens / inference_time
    print(f"Tokens per second: {tokens_per_second}")

    return {"generated_text": generated_text}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
