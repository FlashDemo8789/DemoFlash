#!/usr/bin/env python3
"""
Generate Pydantic schema and TypeScript types from metrics.json.

This script reads the metrics.json file from backend/contracts and generates:
1. Backend Pydantic schema (backend/app/schemas.py)
2. Frontend TypeScript types (frontend/types/metrics.ts)
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List

# File paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
METRICS_JSON_PATH = ROOT_DIR / "flashcamp" / "backend" / "contracts" / "metrics.json"
PYDANTIC_OUTPUT_PATH = ROOT_DIR / "flashcamp" / "backend" / "app" / "schemas.py"
TYPESCRIPT_OUTPUT_PATH = ROOT_DIR / "flashcamp" / "frontend" / "types" / "metrics.ts"

# Ensure output directories exist
PYDANTIC_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
TYPESCRIPT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_metrics() -> List[Dict[str, Any]]:
    """Load metrics from JSON file"""
    with open(METRICS_JSON_PATH, 'r') as f:
        return json.load(f)

def generate_pydantic_schema(metrics: List[Dict[str, Any]]) -> str:
    """Generate Pydantic schema from metrics"""
    lines = [
        "# AUTO-GENERATED from contracts/metrics.json - DO NOT EDIT MANUALLY",
        "from typing import List, Any, Dict, Optional, Union",
        "from pydantic import BaseModel, Field",
        "",
        "",
        "class MetricsInput(BaseModel):",
        "    \"\"\"Input model for startup metrics\"\"\"",
    ]
    
    # Add fields to the Pydantic model
    for metric in metrics:
        key = metric["key"]
        label = metric["label"]
        metric_type = metric["type"]
        default = metric.get("default")
        
        # Determine Python type based on metric type
        if metric_type == "number":
            py_type = "float"
        elif metric_type == "checkbox":
            py_type = "bool"
        elif metric_type == "list":
            py_type = "List[str]"
        elif metric_type == "any":
            py_type = "Any"
        else:  # Default to string for text/string type
            py_type = "str"
        
        # Format the field
        field_str = f"    {key}: Optional[{py_type}] = Field(default={repr(default)}"
        
        # Add description if needed
        field_str += f", description=\"{label}\""
        
        field_str += ")"
        lines.append(field_str)
    
    # Add response models
    lines.extend([
        "",
        "",
        "class Alert(BaseModel):",
        "    \"\"\"Model for alerts\"\"\"",
        "    type: str",
        "    message: str",
        "    severity: str = \"warning\"  # info, warning, error",
        "",
        "",
        "class AnalysisResult(BaseModel):",
        "    \"\"\"Result of startup analysis\"\"\"",
        "    pillar_scores: Dict[str, float]",
        "    overall_score: float",
        "    success_probability: float",
        "    alerts: List[Alert] = []",
        "",
        "",
        "class RunwaySimulation(BaseModel):",
        "    \"\"\"Model for runway simulation results\"\"\"",
        "    dates: List[str]",
        "    capital: List[float]",
        "    runway_months: int",
        "",
        "",
        "class PortfolioSimulation(BaseModel):",
        "    \"\"\"Model for portfolio simulation results\"\"\"",
        "    companies: List[str]",
        "    scores: Dict[str, Dict[str, float]]",
    ])
    
    return "\n".join(lines)

def generate_typescript_types(metrics: List[Dict[str, Any]]) -> str:
    """Generate TypeScript types from metrics"""
    lines = [
        "// AUTO-GENERATED from contracts/metrics.json - DO NOT EDIT MANUALLY",
        "",
        "// Metric contract for a single metric",
        "export interface MetricContract {",
        "  key: string;",
        "  label: string;",
        "  type: 'text' | 'number' | 'checkbox' | 'list';",
        "  default: any;",
        "  pillar: 'Capital' | 'Advantage' | 'Market' | 'People' | 'Info';",
        "}",
        "",
        "// Input model for all metrics",
        "export interface MetricsInput {",
    ]
    
    # Add fields to the TypeScript interface
    for metric in metrics:
        key = metric["key"]
        metric_type = metric["type"]
        
        # Determine TypeScript type based on metric type
        if metric_type == "number":
            ts_type = "number"
        elif metric_type == "checkbox":
            ts_type = "boolean"
        elif metric_type == "list":
            ts_type = "string[]"
        else:  # Default to string for text/string type
            ts_type = "string"
        
        # Fields are optional
        field_str = f"  {key}?: {ts_type};"
        lines.append(field_str)
    
    # Close the interface
    lines.append("}")
    
    # Add response models
    lines.extend([
        "",
        "// Alert model",
        "export interface Alert {",
        "  type: string;",
        "  message: string;",
        "  severity: 'info' | 'warning' | 'error';",
        "}",
        "",
        "// Analysis result model",
        "export interface AnalysisResult {",
        "  pillar_scores: Record<string, number>;",
        "  overall_score: number;",
        "  success_probability: number;",
        "  alerts: Alert[];",
        "}",
        "",
        "// Runway simulation model",
        "export interface RunwaySimulation {",
        "  dates: string[];",
        "  capital: number[];",
        "  runway_months: number;",
        "}",
        "",
        "// Portfolio simulation model",
        "export interface PortfolioSimulation {",
        "  companies: string[];",
        "  scores: Record<string, Record<string, number>>;",
        "}",
        "",
        "// Helper to get metrics by pillar",
        "export const getMetricsByPillar = (metrics: MetricContract[], pillar: string): MetricContract[] => {",
        "  return metrics.filter(m => m.pillar === pillar);",
        "};",
    ])
    
    return "\n".join(lines)

def main():
    """Main function"""
    print(f"Loading metrics from {METRICS_JSON_PATH}")
    metrics = load_metrics()
    print(f"Loaded {len(metrics)} metrics")
    
    # Generate Pydantic schema
    pydantic_schema = generate_pydantic_schema(metrics)
    with open(PYDANTIC_OUTPUT_PATH, 'w') as f:
        f.write(pydantic_schema)
    print(f"Generated Pydantic schema at {PYDANTIC_OUTPUT_PATH}")
    
    # Generate TypeScript types
    typescript_types = generate_typescript_types(metrics)
    with open(TYPESCRIPT_OUTPUT_PATH, 'w') as f:
        f.write(typescript_types)
    print(f"Generated TypeScript types at {TYPESCRIPT_OUTPUT_PATH}")
    
    print("Done! ✅")

if __name__ == "__main__":
    main() 