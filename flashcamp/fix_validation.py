#!/usr/bin/env python3
"""
Fix for the field name mismatch in the validation logic.
This script will patch the validation.py file to map between 
the Pydantic model field names and the validation expected field names.
"""

import os
import re
import shutil

# Define the mapping between Pydantic fields and validation fields
FIELD_MAPPING = {
    "team_size_total": "team_size",
    "monthly_burn_usd": "burn_rate_monthly", 
    "burn_rate_usd": "burn_rate_monthly",
    "cash_on_hand_usd": "cash_on_hand",
    # Add more mappings as needed
}

def add_field_mapping_code(validation_file):
    """
    Add field mapping code to the validation_metrics_input function
    """
    with open(validation_file, 'r') as f:
        content = f.read()
    
    # Create a backup
    backup_file = validation_file + '.bak'
    shutil.copy2(validation_file, backup_file)
    print(f"Created backup at {backup_file}")
    
    # Find the validate_metrics_input function
    match = re.search(r'def validate_metrics_input\(data: Dict\[str, Any\]\) -> ValidationResult:.*?    # Start with required fields validation', content, re.DOTALL)
    if not match:
        print("Could not find the validate_metrics_input function, aborting.")
        return False
    
    # Prepare the mapping code to insert
    mapping_code = """def validate_metrics_input(data: Dict[str, Any]) -> ValidationResult:
    \"\"\"
    Validate metrics input data with comprehensive checks
    Returns ValidationResult with valid flag and errors list
    \"\"\"
    # Map field names from Pydantic model to validation expected names
    field_mapping = {
        "team_size_total": "team_size",
        "monthly_burn_usd": "burn_rate_monthly", 
        "burn_rate_usd": "burn_rate_monthly",
        "cash_on_hand_usd": "cash_on_hand",
    }
    
    # Apply field mapping
    for pydantic_field, validation_field in field_mapping.items():
        if pydantic_field in data and data[pydantic_field] is not None:
            data[validation_field] = data[pydantic_field]
    
    # Start with required fields validation"""
    
    # Replace the function start with our modified version
    modified_content = content.replace(match.group(0), mapping_code)
    
    # Write the modified content back
    with open(validation_file, 'w') as f:
        f.write(modified_content)
    
    print(f"Successfully modified {validation_file}")
    return True

if __name__ == "__main__":
    # Path to validation.py
    validation_file = "backend/validation.py"
    
    if not os.path.exists(validation_file):
        print(f"Validation file not found at {validation_file}")
        exit(1)
    
    if add_field_mapping_code(validation_file):
        print("Fix applied successfully. Restart the server for changes to take effect.")
    else:
        print("Failed to apply fix.") 