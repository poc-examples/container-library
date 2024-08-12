# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import multiprocessing
# import logging
# import os
# from multiprocessing_server import worker_process
# from config import NUM_WORKERS

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class TextRequest(BaseModel):
#     input_text: str

# app = FastAPI()

# # Configure CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

# request_queue = multiprocessing.Queue()
# response_queue = multiprocessing.Queue()

# workers = []
# for _ in range(NUM_WORKERS):
#     p = multiprocessing.Process(target=worker_process, args=(request_queue, response_queue))
#     p.start()
#     workers.append(p)

# @app.post("/generate-text")
# async def generate_text(request: TextRequest):
#     prompt = request.input_text
#     request_queue.put(prompt)
#     response_text = response_queue.get()
#     return {"generated_text": f"{response_text}"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8080)

#     # Clean up workers on shutdown
#     for _ in workers:
#         request_queue.put(None)
#     for p in workers:
#         p.join()
