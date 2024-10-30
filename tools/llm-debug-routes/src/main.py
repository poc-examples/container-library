from fastapi import FastAPI, HTTPException, Request
import logging
import json

app = FastAPI()

logging.basicConfig(level=logging.INFO)

# Middleware to log any incoming request
@app.middleware("http")
async def log_requests(request: Request, call_next):
    body = await request.body()
    logging.info(f"Request method: {request.method} | URL: {request.url} | Body: {json.dumps(body.decode('utf-8')) if body else 'No body'}")
    
    # Proceed to the next middleware or route
    response = await call_next(request)
    return response

# General success response for all routes
async def log_request_and_return_success(request: Request):
    body = await request.json() if request.method in ["POST", "PUT"] else None
    logging.info(f"Request data: {body}")
    return {"status": "success", "code": 200}