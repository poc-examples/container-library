from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langflow.load import run_flow_from_json, load_flow_from_json

app = FastAPI()
flow = "../flows/rag_chat_indexer.json"

runner = load_flow_from_json(flow=flow)

class FlowRequest(BaseModel):
    input_value: str

class FlowResponse(BaseModel):
    response: str

@app.post("/chat", response_model=FlowResponse)
async def run_flow_endpoint(request: FlowRequest):
    try:
        response = run_flow_from_json(
            flow=flow, 
            input_value=""
        )

        return FlowResponse()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/healthz")
async def read_root():
    return {"message": "LangFlow FastAPI is running!"}
