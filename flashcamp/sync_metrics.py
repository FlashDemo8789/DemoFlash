#!/usr/bin/env python
"""
Enhanced Metrics Synchronization Tool

This script ensures metrics consistency across the entire FLASH system:
1. Syncs frontend metrics.json with CSV headers (CSV as source of truth)
2. Updates TypeScript type definitions for the frontend
3. Synchronizes backend Pydantic models
4. Validates metrics usage in key components

Run this script whenever the metrics definitions change or when inconsistencies are suspected.
"""
import os
import sys
import json
import csv
import logging
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Project paths
ROOT = Path(__file__).parent.absolute()
CSV_PATH = ROOT / "data" / "seed_dataset_master_final_54000_68.csv"
FRONTEND_METRICS_PATH = ROOT / "frontend" / "constants" / "metrics.json"
CONSTANTS_METRICS_PATH = ROOT / "constants" / "metrics.py"
TYPESCRIPT_TYPES_PATH = ROOT / "frontend" / "types" / "metrics.ts"
BACKEND_SCHEMA_PATH = ROOT / "backend" / "schemas.py"
CONTRACTS_PATH = ROOT / "contracts" / "metrics.json"

def get_csv_headers() -> List[str]:
    """Extract headers from the CSV file (source of truth)"""
    try:
        with open(CSV_PATH, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            return headers
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        sys.exit(1)

def get_frontend_metrics() -> List[Dict[str, Any]]:
    """Load the current frontend metrics"""
    try:
        with open(FRONTEND_METRICS_PATH, 'r') as f:
            metrics = json.load(f)
            return metrics
    except Exception as e:
        logger.error(f"Error reading frontend metrics: {e}")
        sys.exit(1)

def save_frontend_metrics(metrics: List[Dict[str, Any]]) -> None:
    """Save updated metrics to the frontend file"""
    try:
        with open(FRONTEND_METRICS_PATH, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"✅ Updated frontend metrics saved to {FRONTEND_METRICS_PATH}")
    except Exception as e:
        logger.error(f"Error saving frontend metrics: {e}")
        sys.exit(1)

def save_contract_metrics(metrics: List[Dict[str, Any]]) -> None:
    """Save metrics to the contracts directory for API consistency"""
    if not CONTRACTS_PATH.parent.exists():
        CONTRACTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(CONTRACTS_PATH, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"✅ Updated contract metrics saved to {CONTRACTS_PATH}")
    except Exception as e:
        logger.error(f"Error saving contract metrics: {e}")

def update_constants_metrics(csv_headers: List[str], frontend_metrics: List[Dict[str, Any]]) -> bool:
    """Update the ALL_METRICS list in constants/metrics.py"""
    try:
        # Read the current file content
        with open(CONSTANTS_METRICS_PATH, 'r') as f:
            content = f.read()
        
        # Extract existing metrics to preserve attributes
        start_marker = "ALL_METRICS = ["
        end_marker = "]"
        start_idx = content.find(start_marker) + len(start_marker)
        end_idx = content.rfind(end_marker, start_idx)
        
        if start_idx == -1 or end_idx == -1:
            logger.error("Could not find ALL_METRICS list in constants/metrics.py")
            return False
        
        # Get the current metrics
        metrics_block = content[start_idx:end_idx].strip()
        
        # Create a new metrics block based on frontend metrics
        new_metrics_lines = []
        for metric in frontend_metrics:
            # Format each metric as it appears in the Python file
            line = f'    {{"name": "{metric["name"]}",'
            # Pad the name to align columns
            line = line.ljust(45)
            
            line += f'"pillar": "{metric["pillar"]}",'
            # Pad the pillar to align columns
            line = line.ljust(65)
            
            line += f'"label": "{metric["label"]}",'
            # Pad the label to align columns
            line = line.ljust(110)
            
            line += f'"type": "{metric["type"]}",'
            # Pad the type to align columns
            line = line.ljust(130)
            
            tip = metric.get("tip", "")
            line += f'"tip": "{tip}"}}'
            
            new_metrics_lines.append(line)
        
        # Combine the lines
        new_metrics_block = ",\n".join(new_metrics_lines)
        
        # Replace the old metrics block with the new one
        new_content = content[:start_idx] + "\n" + new_metrics_block + "\n" + content[end_idx:]
        
        # Only write if there are actual changes
        if new_metrics_block != metrics_block:
            with open(CONSTANTS_METRICS_PATH, 'w') as f:
                f.write(new_content)
            logger.info(f"✅ Updated constants/metrics.py with {len(frontend_metrics)} metrics")
            return True
        else:
            logger.info("✅ constants/metrics.py already up-to-date")
            return False
        
    except Exception as e:
        logger.error(f"Error updating constants metrics: {e}")
        return False

def update_typescript_types() -> bool:
    """Update TypeScript type definitions based on metrics.json"""
    try:
        # Generate TypeScript interface from metrics.json
        metrics = get_frontend_metrics()
        
        # Create a set of all metric types
        metric_types = set(m["type"] for m in metrics)
        
        # Map metric types to TypeScript types
        ts_type_map = {
            "text": "string",
            "number": "number",
            "checkbox": "boolean",
            "list": "string[]",
            "select": "string"
        }
        
        # Create TypeScript content
        ts_content = [
            "/**",
            " * AUTO-GENERATED TypeScript interfaces for metrics",
            " * Generated by sync_metrics.py",
            " * DO NOT EDIT DIRECTLY",
            " */",
            "",
            "// Enum of all available metric names",
            "export enum MetricName {",
        ]
        
        # Add metric name enum values
        for metric in metrics:
            ts_content.append(f'  {metric["name"]} = "{metric["name"]}",')
        
        ts_content.append("}")
        ts_content.append("")
        
        # Add pillar enum
        pillars = sorted(set(m["pillar"] for m in metrics))
        ts_content.append("// Enum of all metric pillars")
        ts_content.append("export enum MetricPillar {")
        for pillar in pillars:
            ts_content.append(f'  {pillar} = "{pillar}",')
        ts_content.append("}")
        ts_content.append("")
        
        # Add metric type enum
        ts_content.append("// Enum of all metric types")
        ts_content.append("export enum MetricType {")
        for metric_type in metric_types:
            ts_content.append(f'  {metric_type} = "{metric_type}",')
        ts_content.append("}")
        ts_content.append("")
        
        # Add metric interface
        ts_content.append("// Individual metric definition")
        ts_content.append("export interface Metric {")
        ts_content.append("  name: MetricName;")
        ts_content.append("  pillar: MetricPillar;")
        ts_content.append("  label: string;")
        ts_content.append("  type: MetricType;")
        ts_content.append("  tip: string;")
        ts_content.append("  options?: string[];")
        ts_content.append("}")
        ts_content.append("")
        
        # Add metric values interface
        ts_content.append("// Interface for metric values")
        ts_content.append("export interface MetricValues {")
        for metric in metrics:
            ts_type = ts_type_map.get(metric["type"], "any")
            ts_content.append(f'  {metric["name"]}?: {ts_type};')
        ts_content.append("}")
        ts_content.append("")
        
        # Add grouping helper type 
        ts_content.append("// Type for metrics grouped by pillar")
        ts_content.append("export type MetricsByPillar = Record<MetricPillar, Metric[]>;")
        
        # Save to file
        ts_file_content = "\n".join(ts_content)
        
        # Check if file exists and if content is different
        needs_update = True
        if TYPESCRIPT_TYPES_PATH.exists():
            with open(TYPESCRIPT_TYPES_PATH, 'r') as f:
                current_content = f.read()
                if current_content == ts_file_content:
                    needs_update = False
        
        if needs_update:
            # Make sure directory exists
            TYPESCRIPT_TYPES_PATH.parent.mkdir(parents=True, exist_ok=True)
            
            with open(TYPESCRIPT_TYPES_PATH, 'w') as f:
                f.write(ts_file_content)
            logger.info(f"✅ Updated TypeScript types at {TYPESCRIPT_TYPES_PATH}")
            return True
        else:
            logger.info("✅ TypeScript types already up-to-date")
            return False
            
    except Exception as e:
        logger.error(f"Error updating TypeScript types: {e}")
        return False

def update_pydantic_model() -> bool:
    """Update or create the backend Pydantic model for metrics"""
    try:
        # Check if backend schema file exists
        if not BACKEND_SCHEMA_PATH.exists():
            logger.warning(f"Backend schemas file not found at {BACKEND_SCHEMA_PATH}")
            logger.warning("Skipping Pydantic model update")
            return False
            
        # Read the current schema file
        with open(BACKEND_SCHEMA_PATH, 'r') as f:
            schema_content = f.read()
            
        # Check if MetricsInput class exists
        metrics_input_match = re.search(r'class\s+MetricsInput\s*\(\s*BaseModel\s*\)\s*:(.*?)(?:class|\Z)', 
                                       schema_content, re.DOTALL)
        
        if not metrics_input_match:
            logger.warning("MetricsInput class not found in backend schemas")
            logger.warning("Will create a new schema class")
            
            # Get metrics from frontend
            metrics = get_frontend_metrics()
            
            # Create a new MetricsInput class
            new_class_lines = [
                "\n\nclass MetricsInput(BaseModel):",
                '    """Input model for startup metrics analysis"""',
                ""
            ]
            
            # Map JSON types to Pydantic types
            pydantic_type_map = {
                "text": "str",
                "number": "Optional[float]",
                "checkbox": "Optional[bool]",
                "list": "Optional[List[str]]",
                "select": "Optional[str]"
            }
            
            # Add field definitions
            for metric in metrics:
                field_type = pydantic_type_map.get(metric["type"], "Any")
                field_name = metric["name"]
                field_desc = metric.get("tip", metric["label"])
                
                # Add field with description and optional
                new_class_lines.append(f'    {field_name}: {field_type} = Field(None, description="{field_desc}")')
            
            # Check for required imports
            needed_imports = [
                "from typing import List, Optional, Any",
                "from pydantic import BaseModel, Field"
            ]
            
            # Add imports if not present
            for import_line in needed_imports:
                if import_line not in schema_content:
                    schema_content = import_line + "\n" + schema_content
            
            # Add the new class to the end of the file
            new_schema_content = schema_content + "\n".join(new_class_lines)
            
            # Write the updated schema
            with open(BACKEND_SCHEMA_PATH, 'w') as f:
                f.write(new_schema_content)
            
            logger.info(f"✅ Created new MetricsInput model in {BACKEND_SCHEMA_PATH}")
            return True
        else:
            # MetricsInput class exists, update the fields
            class_content = metrics_input_match.group(1)
            
            # Get metrics from frontend
            metrics = get_frontend_metrics()
            
            # Check which fields are missing
            field_pattern = r'^\s*(\w+)\s*:'
            existing_fields = re.findall(field_pattern, class_content, re.MULTILINE)
            existing_fields_set = set(existing_fields)
            
            metric_names = [m["name"] for m in metrics]
            metric_names_set = set(metric_names)
            
            missing_fields = metric_names_set - existing_fields_set
            extra_fields = existing_fields_set - metric_names_set
            
            if not missing_fields and not extra_fields:
                logger.info("✅ Backend Pydantic model already up-to-date")
                return False
            
            # We need to update the model
            logger.info(f"Updating Pydantic model, adding {len(missing_fields)} fields, removing {len(extra_fields)} fields")
            
            # Create new class content
            pydantic_type_map = {
                "text": "str",
                "number": "Optional[float]",
                "checkbox": "Optional[bool]",
                "list": "Optional[List[str]]",
                "select": "Optional[str]"
            }
            
            # Extract class definition line
            class_def_pattern = r'(class\s+MetricsInput\s*\(\s*BaseModel\s*\)\s*:.*?\n)(.*)(?:class|\Z)'
            class_match = re.search(class_def_pattern, schema_content, re.DOTALL)
            
            if not class_match:
                logger.error("Failed to parse MetricsInput class structure")
                return False
                
            class_def_line = class_match.group(1)
            
            # Create new class lines
            new_class_lines = [
                class_def_line,
                '    """Input model for startup metrics analysis"""',
                ""
            ]
            
            # Add field definitions for all metrics
            for metric in metrics:
                field_type = pydantic_type_map.get(metric["type"], "Any")
                field_name = metric["name"]
                field_desc = metric.get("tip", metric["label"])
                
                # Add field with description and optional
                new_class_lines.append(f'    {field_name}: {field_type} = Field(None, description="{field_desc}")')
            
            # Check for required imports
            needed_imports = [
                "from typing import List, Optional, Any",
                "from pydantic import BaseModel, Field"
            ]
            
            # Replace the old class with the new one
            new_schema_content = re.sub(
                r'class\s+MetricsInput\s*\(\s*BaseModel\s*\)\s*:.*?(?=class|\Z)', 
                "\n".join(new_class_lines), 
                schema_content,
                flags=re.DOTALL
            )
            
            # Add imports if not present
            for import_line in needed_imports:
                if import_line not in new_schema_content:
                    new_schema_content = import_line + "\n" + new_schema_content
            
            # Write the updated schema
            with open(BACKEND_SCHEMA_PATH, 'w') as f:
                f.write(new_schema_content)
            
            logger.info(f"✅ Updated MetricsInput model in {BACKEND_SCHEMA_PATH}")
            return True
            
    except Exception as e:
        logger.error(f"Error updating Pydantic model: {e}")
        return False

def check_component_usage() -> Dict[str, Set[str]]:
    """Check which metrics are used in key application components"""
    # Key components to check
    components = [
        ROOT / "frontend" / "src" / "components" / "ResultsPage.tsx",
        ROOT / "frontend" / "src" / "components" / "WizardPage.tsx",
        ROOT / "frontend" / "src" / "components" / "PillarStep.tsx",
        ROOT / "frontend" / "src" / "components" / "PDFDownloadButton.tsx",
        ROOT / "backend" / "app" / "engines" / "ml.py",
    ]
    
    usage = {}
    metrics_names = [m["name"] for m in get_frontend_metrics()]
    
    for component_path in components:
        if not component_path.exists():
            continue
            
        component_name = str(component_path.relative_to(ROOT))
        usage[component_name] = set()
        
        try:
            with open(component_path, 'r') as f:
                content = f.read()
                
            # Check for each metric
            for metric_name in metrics_names:
                if metric_name in content:
                    usage[component_name].add(metric_name)
                    
            logger.info(f"Found {len(usage[component_name])} metrics used in {component_name}")
            
        except Exception as e:
            logger.error(f"Error checking {component_name}: {e}")
    
    return usage

def generate_usage_report(usage: Dict[str, Set[str]]) -> None:
    """Generate a report of metrics usage across components"""
    metrics = get_frontend_metrics()
    metrics_dict = {m["name"]: m for m in metrics}
    
    # Compile all metrics that are used in any component
    used_metrics = set()
    for component, metric_set in usage.items():
        used_metrics.update(metric_set)
    
    # Find metrics not used in any component
    unused_metrics = set(metrics_dict.keys()) - used_metrics
    
    # Generate report
    report_lines = [
        "# Metrics Usage Report",
        "",
        f"Total metrics: {len(metrics)}",
        f"Metrics used in components: {len(used_metrics)}",
        f"Metrics not used in any component: {len(unused_metrics)}",
        "",
        "## Usage by Component",
        ""
    ]
    
    for component, metric_set in usage.items():
        report_lines.append(f"### {component}")
        report_lines.append(f"Uses {len(metric_set)} metrics:")
        report_lines.append("")
        
        # Group by pillar
        metrics_by_pillar = {}
        for metric_name in metric_set:
            if metric_name in metrics_dict:
                pillar = metrics_dict[metric_name]["pillar"]
                if pillar not in metrics_by_pillar:
                    metrics_by_pillar[pillar] = []
                metrics_by_pillar[pillar].append(metric_name)
        
        # Show metrics by pillar
        for pillar, pillar_metrics in sorted(metrics_by_pillar.items()):
            report_lines.append(f"**{pillar}**: {', '.join(sorted(pillar_metrics))}")
            
        report_lines.append("")
    
    # List unused metrics
    if unused_metrics:
        report_lines.append("## Unused Metrics")
        report_lines.append("")
        
        # Group by pillar
        unused_by_pillar = {}
        for metric_name in unused_metrics:
            if metric_name in metrics_dict:
                pillar = metrics_dict[metric_name]["pillar"]
                if pillar not in unused_by_pillar:
                    unused_by_pillar[pillar] = []
                unused_by_pillar[pillar].append(metric_name)
        
        # Show unused metrics by pillar
        for pillar, pillar_metrics in sorted(unused_by_pillar.items()):
            report_lines.append(f"**{pillar}**: {', '.join(sorted(pillar_metrics))}")
    
    # Save report
    report_path = ROOT / "metrics_usage_report.md"
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
    
    logger.info(f"✅ Generated metrics usage report at {report_path}")

def propagate_changes(updated_frontend_metrics: bool, updated_constants: bool) -> None:
    """Propagate changes to other parts of the system if needed"""
    if updated_frontend_metrics or updated_constants:
        # Run build script if it exists
        build_script = ROOT / "generators" / "build_schema.py"
        if build_script.exists():
            logger.info("Running build_schema.py to propagate changes...")
            try:
                result = subprocess.run(
                    [sys.executable, str(build_script)],
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    logger.info("✅ Successfully ran build_schema.py")
                    logger.info(result.stdout)
                else:
                    logger.error(f"Error running build_schema.py: {result.stderr}")
            except Exception as e:
                logger.error(f"Failed to run build_schema.py: {e}")
        
        # Notify about other recommended actions
        logger.info("\n=== NEXT STEPS ===")
        logger.info("You may need to:")
        logger.info("1. Restart the backend (if running)")
        logger.info("2. Rebuild the frontend (npm run build)")
        logger.info("3. Review any type errors in the frontend code")

def sync_metrics() -> Dict[str, Any]:
    """Synchronize metrics across all system components"""
    # Get headers from CSV (source of truth)
    csv_headers = get_csv_headers()
    logger.info(f"Found {len(csv_headers)} headers in CSV file")
    
    # Get current frontend metrics
    frontend_metrics = get_frontend_metrics()
    logger.info(f"Found {len(frontend_metrics)} metrics in frontend file")
    
    # Create a map of existing frontend metrics for easy lookup
    frontend_metrics_map = {m["name"]: m for m in frontend_metrics}
    
    # Check for metrics in CSV but missing from frontend
    missing_in_frontend = [h for h in csv_headers if h not in frontend_metrics_map]
    if missing_in_frontend:
        logger.warning(f"Found {len(missing_in_frontend)} metrics in CSV missing from frontend")
        for metric in missing_in_frontend:
            logger.info(f"Adding missing metric: {metric}")
            # Create a default metric entry
            new_metric = {
                "name": metric,
                "pillar": "Info",  # Default pillar
                "label": " ".join(w.capitalize() for w in metric.split('_')),
                "type": "number" if any(term in metric for term in ["count", "ratio", "percent", "score", "rate", "months", "years", "days", "usd", "size"]) else "text",
                "tip": f"Data for {' '.join(metric.split('_'))}"
            }
            # Special handling for certain field types
            if "present" in metric or metric.startswith("has_"):
                new_metric["type"] = "checkbox"
            elif "list" in metric or metric.endswith("flags") or metric.endswith("barriers"):
                new_metric["type"] = "list"
            
            frontend_metrics.append(new_metric)
    
    # Check for metrics in frontend but not in CSV
    extra_in_frontend = [m["name"] for m in frontend_metrics if m["name"] not in csv_headers]
    if extra_in_frontend:
        logger.warning(f"Found {len(extra_in_frontend)} metrics in frontend not in CSV")
        # Remove extra metrics or mark them (we'll just log them in this version)
        for metric in extra_in_frontend:
            logger.info(f"Extra metric: {metric}")
    
    # Save updated frontend metrics if changes were made
    updated_frontend = False
    if missing_in_frontend:
        save_frontend_metrics(frontend_metrics)
        save_contract_metrics(frontend_metrics)
        logger.info(f"✅ Added {len(missing_in_frontend)} missing metrics to frontend")
        updated_frontend = True
    else:
        logger.info("✅ All CSV metrics exist in frontend metrics.json")
    
    # Update constants/metrics.py
    updated_constants = update_constants_metrics(csv_headers, frontend_metrics)
    
    # Update TypeScript types
    updated_ts = update_typescript_types()
    
    # Update Pydantic model
    updated_pydantic = update_pydantic_model()
    
    # Check component usage
    usage = check_component_usage()
    generate_usage_report(usage)
    
    # Propagate changes to other parts of the system
    propagate_changes(updated_frontend, updated_constants)
    
    return {
        "csv_headers_count": len(csv_headers),
        "frontend_metrics_count": len(frontend_metrics),
        "missing_in_frontend": missing_in_frontend,
        "extra_in_frontend": extra_in_frontend,
        "updated_frontend": updated_frontend,
        "updated_constants": updated_constants,
        "updated_typescript": updated_ts,
        "updated_pydantic": updated_pydantic
    }

def main():
    """Main function to synchronize metrics across the system"""
    logger.info("Starting enhanced metrics synchronization...")
    results = sync_metrics()
    
    # Print summary
    logger.info("\n=== SYNCHRONIZATION SUMMARY ===")
    logger.info(f"CSV Headers: {results['csv_headers_count']}")
    logger.info(f"Frontend Metrics: {results['frontend_metrics_count']}")
    
    if results["missing_in_frontend"]:
        logger.info(f"✅ Added {len(results['missing_in_frontend'])} missing metrics to frontend")
    if results["extra_in_frontend"]:
        logger.warning(f"⚠️ Found {len(results['extra_in_frontend'])} metrics in frontend not in CSV")
        
    logger.info(f"Updated constants/metrics.py: {results['updated_constants']}")
    logger.info(f"Updated TypeScript types: {results['updated_typescript']}")
    logger.info(f"Updated Pydantic model: {results['updated_pydantic']}")
    
    # Final status
    if results["missing_in_frontend"] or results["updated_constants"] or results["updated_typescript"] or results["updated_pydantic"]:
        logger.info("\n✅ SUCCESS: Metrics synchronized across all system components")
        return 0
    else:
        logger.info("\n✅ SUCCESS: All metrics already synchronized")
        return 0

def test_backend_schema():
    """Test backend schema compatibility with a minimal payload"""
    import requests
    import json
    
    # Minimal payload with all required fields
    test_payload = {
        "startup_id": "test-123",
        "startup_name": "Test Startup",
        "funding_stage": "Seed",
        "team_size_full_time": 5,
        "monthly_burn_usd": 50000,
        "revenue_monthly_usd": 10000,
        "cash_on_hand_usd": 500000,
        "runway_months": 10,
        "market_growth_rate_percent": 15,
        "founder_domain_experience_years": 7
    }
    
    print(f"\nTesting backend API with minimal payload...")
    try:
        response = requests.post(
            "http://localhost:8000/api/analyze",
            json=test_payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print("✅ Backend API validation passed!")
            return True
        else:
            print(f"❌ Backend API validation failed: {response.status_code}")
            print(f"Error details: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing backend API: {str(e)}")
        return False

# Run the test if executed directly
if __name__ == "__main__":
    sync_metrics()
    test_backend_schema()
