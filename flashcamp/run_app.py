import sys
import os
import importlib.util

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try different import approaches and show debug info
print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")
print(f"Contents of backend directory:")
for item in os.listdir("backend"):
    print(f"  - {item}")

print("Attempting import directly from file path...")
# Direct import from file path
try:
    spec = importlib.util.spec_from_file_location("app_module", "backend/app.py")
    app_module = importlib.util.module_from_spec(spec)
    sys.modules["app_module"] = app_module
    spec.loader.exec_module(app_module)
    app = app_module.app
    print("Successfully imported app!")
    print(f"App routes: {[route.path for route in app.routes]}")
except Exception as e:
    print(f"Import error: {e}")
    raise

# Can be run with: uvicorn run_app:app --host 0.0.0.0 --port 8000 