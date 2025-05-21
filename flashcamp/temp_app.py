"""
Temporary test app to bypass validation
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import random
import json

app = FastAPI()

@app.get("/")
async def root():
    return {
        "message": "Test app running",
        "status": "healthy"
    }

@app.post("/api/analyze")
async def analyze(request: Request):
    """Analyze startup metrics (test bypass validation)"""
    try:
        # Get the raw JSON payload
        data = await request.json()
        
        print(f"Received data: {json.dumps(data, indent=2)}")
        
        # Generate random scores
        pillar_scores = {
            "Market": round(random.uniform(0.5, 0.9), 2),
            "Advantage": round(random.uniform(0.5, 0.9), 2),
            "People": round(random.uniform(0.5, 0.9), 2),
            "Capital": round(random.uniform(0.5, 0.9), 2),
        }
        
        # Calculate overall score
        overall_score = sum(pillar_scores.values()) / len(pillar_scores)
        
        # Generate random success probability
        success_prob = round(random.uniform(0.3, 0.8), 2)
        
        # Return mock response
        return {
            "pillar_scores": pillar_scores,
            "overall_score": overall_score,
            "alerts": [],
            "success_probability": success_prob
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        ) 