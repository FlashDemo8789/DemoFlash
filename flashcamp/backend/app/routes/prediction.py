#!/usr/bin/env python
"""
FastAPI routes for startup success prediction using the hierarchical model.
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse

# Import from the enhanced ML engine instead of a separate implementation
from flashcamp.backend.app.engines.ml import (
    predict_success_probability,
    generate_recommendations,
    get_model_metadata
)
from flashcamp.backend.validation import sanitize_input
from flashcamp.backend.app.utils.visualize import generate_prediction_chart

router = APIRouter(
    prefix="/api/prediction",
    tags=["prediction"],
    responses={404: {"description": "Not found"}},
)

# Pydantic models for request/response
class PredictionResponse(BaseModel):
    pillar_scores: Dict[str, float]
    final_score: float
    prediction: str
    confidence: float
    threshold: float

class RecommendationItem(BaseModel):
    metric: str
    recommendation: str
    impact: str

class RecommendationsResponse(BaseModel):
    capital: List[RecommendationItem] = Field(default_factory=list)
    advantage: List[RecommendationItem] = Field(default_factory=list)
    market: List[RecommendationItem] = Field(default_factory=list)
    people: List[RecommendationItem] = Field(default_factory=list)

class ModelMetrics(BaseModel):
    auc: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    calibration_error: Optional[float] = None

class ModelInfoResponse(BaseModel):
    model_version: str
    dataset_size: int
    success_rate: float
    threshold: float
    pillar_metrics: Dict[str, ModelMetrics] = Field(default_factory=dict)
    meta_metrics: ModelMetrics = Field(default_factory=ModelMetrics)

@router.post("/predict", response_model=PredictionResponse)
async def predict_startup(startup_data: Dict[str, Any]):
    """
    Predict the success probability of a startup using the hierarchical model.
    
    Returns:
        Prediction results with pillar scores and final score
    """
    # Sanitize input data
    clean_data = sanitize_input(startup_data)
    
    try:
        # Run prediction
        result = predict_success_probability(clean_data)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )

@router.post("/recommendations", response_model=RecommendationsResponse)
async def get_recommendations(startup_data: Dict[str, Any]):
    """
    Get recommendations for improving startup success probability.
    
    Returns:
        Recommendations for each pillar based on the startup's metrics
    """
    # Sanitize input data
    clean_data = sanitize_input(startup_data)
    
    try:
        # Get recommendations
        raw_recommendations = generate_recommendations(clean_data)
        
        # Convert to Pydantic model format
        recommendations = {}
        for pillar, recs in raw_recommendations.items():
            recommendations[pillar] = [
                RecommendationItem(
                    metric=rec["metric"],
                    recommendation=rec["recommendation"],
                    impact=rec["impact"]
                )
                for rec in recs
            ]
        
        return RecommendationsResponse(**recommendations)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )

@router.post("/visualization")
async def get_prediction_visualization(startup_data: Dict[str, Any]):
    """
    Generate a visualization of the prediction results.
    
    Returns:
        Visualization image as PNG
    """
    # Sanitize input data
    clean_data = sanitize_input(startup_data)
    
    try:
        # Run prediction
        prediction_result = predict_success_probability(clean_data)
        
        # Generate visualization
        return generate_prediction_chart(prediction_result, clean_data)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating visualization: {str(e)}"
        )

@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_information():
    """
    Get information about the hierarchical model.
    
    Returns:
        Model metadata and performance metrics
    """
    try:
        info = get_model_metadata()
        
        # Convert metrics dictionaries to ModelMetrics objects
        pillar_metrics = {}
        for pillar, metrics in info.get('pillar_metrics', {}).items():
            pillar_metrics[pillar] = ModelMetrics(**metrics)
        
        meta_metrics = ModelMetrics(**info.get('meta_metrics', {}))
        
        # Create response
        response = {
            "model_version": info.get('model_version', 'unknown'),
            "dataset_size": info.get('dataset_size', 0),
            "success_rate": info.get('success_rate', 0),
            "threshold": info.get('threshold', 0.5),
            "pillar_metrics": pillar_metrics,
            "meta_metrics": meta_metrics
        }
        
        return ModelInfoResponse(**response)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model information: {str(e)}"
        ) 