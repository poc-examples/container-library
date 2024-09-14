from fastapi import FastAPI, HTTPException
from typing import List
import os

app = FastAPI()

@app.post("/fetch")
async def fetch(request):
    # Check Payloads
    print(request["request"])

    # Initialize a dictionary to store the results
    results = {}
    
    for var_name in request["request"]:
        value = os.getenv(var_name)
        
        # If the environment variable doesn't exist, indicate that it's not found
        if value is None:
            results[var_name] = "Not Found"
        else:
            results[var_name] = value
    
    # Return the results as a dictionary
    return results
