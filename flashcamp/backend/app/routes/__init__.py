"""
API routes initialization.
This file collects all API routes from different modules.
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from ..engines import ml
from ...database import get_db
from ...schemas import MetricsInput, RunwaySimInput, AnalysisResult

# Create main router
router = APIRouter()

@router.post("/analyze", response_model=AnalysisResult)
async def analyze(metrics: MetricsInput, db: Session = Depends(get_db)):
    """
    Main analysis endpoint – returns pillar scores, success probability,
    and potential alerts.
    """
    try:
        # Calculate pillar scores
        capital_score = ml.calculate_capital_score(metrics)
        advantage_score = ml.calculate_advantage_score(metrics)
        market_score = ml.calculate_market_score(metrics)
        people_score = ml.calculate_people_score(metrics)
        
        # Calculate overall score (weighted average)
        overall_score = (capital_score * 0.3 + 
                        advantage_score * 0.2 + 
                        market_score * 0.25 + 
                        people_score * 0.25)
        
        # Get success probability
        success_probability = ml.predict_success_probability(metrics)
        
        # Generate alerts based on imbalances
        alerts = []
        if capital_score < 0.3:
            alerts.append({"type": "warning", "pillar": "capital", "message": "Capital score is critically low"})
        if advantage_score < 0.3:
            alerts.append({"type": "warning", "pillar": "advantage", "message": "Advantage score is critically low"})
        if market_score < 0.3:
            alerts.append({"type": "warning", "pillar": "market", "message": "Market score is critically low"})
        if people_score < 0.3:
            alerts.append({"type": "warning", "pillar": "people", "message": "People score is critically low"})
            
        # Check if scores are balanced
        scores = [capital_score, advantage_score, market_score, people_score]
        if max(scores) - min(scores) > 0.4:
            alerts.append({"type": "info", "pillar": "balance", "message": "Large imbalance between pillar scores detected"})
            
        # Return the comprehensive analysis result
        return AnalysisResult(
            startup_id=metrics.startup_id,
            startup_name=metrics.startup_name,
            capital_score=capital_score,
            advantage_score=advantage_score,
            market_score=market_score,
            people_score=people_score,
            overall_score=overall_score,
            success_probability=success_probability,
            alerts=alerts
        )
            
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error during analysis: {str(e)}")

@router.post("/runway")
async def runway_simulation(input_data: RunwaySimInput):
    """
    Simulate runway based on burn rate and capital
    """
    try:
        # Basic calculation: runway = cash / burn
        if input_data.monthly_burn_usd <= 0:
            raise HTTPException(status_code=422, detail="Monthly burn rate must be greater than zero")
            
        runway_months = float(input_data.cash_on_hand_usd / input_data.monthly_burn_usd)
        
        # Generate month-by-month projection
        from datetime import datetime, timedelta
        
        months_to_project = min(int(runway_months * 1.5), 24)  # Project 1.5x the runway or max 24 months
        months_to_project = max(months_to_project, 12)  # At least 12 months
        
        dates = [(datetime.now() + timedelta(days=30*i)).strftime("%Y-%m-%d") for i in range(months_to_project)]
        
        # Calculate remaining capital each month
        initial_capital = float(input_data.cash_on_hand_usd)
        burn = float(input_data.monthly_burn_usd)
        capital_values = [max(0, initial_capital - (burn * i)) for i in range(months_to_project)]
        
        return {
            "runway_months": runway_months,
            "dates": dates,
            "capital": capital_values
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in runway simulation: {str(e)}") 