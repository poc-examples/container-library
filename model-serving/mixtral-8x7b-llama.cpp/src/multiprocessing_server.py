# import multiprocessing
# import logging
# from llama_cpp import Llama
# from huggingface_hub import hf_hub_download
# import os
# from config import NUM_WORKERS

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# model_path = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
# filename = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
# quantization = "models--TheBloke--Mistral-7B-Instruct-v0.2-GGUF/snapshots/3a6fbf4a41a1d52e415a4958cde6856d34b2db93/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
# save_directory = "/ai-models/mistralai"

# logger.info(f"Model Path: {model_path}")
# logger.info(f"Download File: {filename}")
# logger.info(f"Quantization: {quantization}")
# logger.info(f"Model Location: {save_directory}")

# logger.info("""
# We are checking to see if the model is available locally.
# If Model is not available it will be downloaded.
# File sizes are large and may take some time to complete.
# """)

# hf_hub_download(
#     repo_id=model_path, 
#     filename=filename,
#     cache_dir=save_directory
# )

# logger.info(f"Checking model files...")
# os.system(f"ls -la {save_directory}/models--TheBloke--Mistral-7B-Instruct-v0.2-GGUF/snapshots")

# class ModelServer:
#     def __init__(self):
#         logger.info("Starting Model...")
#         try:
#             logger.info(f"Initializing Llama with model path: {save_directory}/{quantization}")
#             self.llm = Llama(
#                 model_path=f"{save_directory}/{quantization}",
#                 n_ctx=28000,  # The max sequence length to use - note that longer sequence lengths require much more resources
#                 n_threads=8,  # The number of CPU threads to use, tailor to your system and the resulting performance
#                 n_gpu_layers=35,
#                 n_copies=1  # Ensure we don't overallocate copies
#             )
#             logger.info("Llama Loaded...")
#         except Exception as e:
#             logger.error("Failed to create llama_context", exc_info=e)
#             raise e

#     def infer(self, prompt):
#         output = self.llm(
#             f"[INST] {prompt} [/INST]", # Prompt
#             max_tokens=28000,  # Generate up to 512 tokens
#             stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
#             echo=False        # Whether to echo the prompt
#         )
#         return output.get("choices")[0]["text"]

# def worker_process(request_queue, response_queue):
#     model_server = ModelServer()
#     while True:
#         prompt = request_queue.get()
#         if prompt is None:  # Use None as a signal to stop the worker
#             break
#         try:
#             response = model_server.infer(prompt)
#             response_queue.put(response)
#         except Exception as e:
#             response_queue.put(f"Error: {str(e)}")

# if __name__ == "__main__":
#     request_queue = multiprocessing.Queue()
#     response_queue = multiprocessing.Queue()

#     workers = []
#     for _ in range(NUM_WORKERS):
#         p = multiprocessing.Process(target=worker_process, args=(request_queue, response_queue))
#         p.start()
#         workers.append(p)

#     # Example to stop workers (place this logic where appropriate in your actual workflow)
#     for _ in workers:
#         request_queue.put(None)
#     for p in workers:
#         p.join()
