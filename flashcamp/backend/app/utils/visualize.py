"""
Visualization utilities for the FlashCAMP predictions.
"""
import io
import numpy as np
import matplotlib.pyplot as plt
from fastapi.responses import StreamingResponse
from typing import Dict, Any

def generate_prediction_chart(prediction_result: Dict[str, Any], startup_data: Dict[str, Any]) -> StreamingResponse:
    """
    Generate a visualization chart for the prediction results.
    
    Args:
        prediction_result: Dictionary with prediction results from the hierarchical model
        startup_data: Dictionary with startup metrics
        
    Returns:
        StreamingResponse with the image data
    """
    # Extract data from prediction result
    pillars = list(prediction_result.get('pillar_scores', {}).keys())
    scores = [prediction_result.get('pillar_scores', {}).get(p, 0.5) for p in pillars]
    final_score = prediction_result.get('final_score', 0.5)
    threshold = prediction_result.get('threshold', 0.5)
    prediction = prediction_result.get('prediction', 'unknown')
    confidence = prediction_result.get('confidence', 0.5)
    
    # Create figure and axes
    plt.figure(figsize=(10, 6))
    
    # Create bar chart for pillar scores
    bar_colors = ['#4CAF50' if s >= 0.5 else '#F44336' for s in scores]
    bars = plt.bar(pillars, scores, color=bar_colors, alpha=0.7)
    
    # Add labels and titles
    plt.ylim(0, 1.0)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    
    # Add final score indicator
    plt.scatter([len(pillars) - 0.5], [final_score], 
                color='blue', s=150, zorder=5, label='Final Score')
    
    # Add threshold line
    plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.5)
    plt.text(len(pillars)/2, threshold + 0.02, f'Threshold ({threshold:.2f})', 
             ha='center', color='r')
    
    # Add annotations
    for i, score in enumerate(scores):
        plt.text(i, score + 0.02, f'{score:.2f}', ha='center')
    
    plt.text(len(pillars) - 0.5, final_score + 0.03, 
             f"{final_score:.2f}", ha='center')
    
    # Add title and styling
    result_text = f"PASS ({confidence:.0%} confident)" if prediction == 'pass' else f"FAIL ({confidence:.0%} confident)"
    plt.title(f"Startup Success Prediction: {result_text}\nSector: {startup_data.get('sector', 'Unknown')}", fontsize=14)
    plt.ylabel('Success Probability')
    plt.grid(axis='y', alpha=0.3)
    
    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    # Return as streaming response
    return StreamingResponse(buf, media_type="image/png") 