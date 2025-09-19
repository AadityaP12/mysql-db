import os

# Base folder inside SIH
base_dir = os.path.join("BACKEND")

# Directory structure
dirs = [
    "app/api",
    "app/core",
    "app/db",
    "app/ml",
    "app/schemas",
    "tests"
]

# Files to create with minimal content
files = {
    # Main entry
    "app/main.py": '''from fastapi import FastAPI
from app.api import routes_auth, routes_data, routes_alerts, routes_ml

app = FastAPI(title="Smart Health Monitoring API")

# Routers
app.include_router(routes_auth.router, prefix="/auth", tags=["Auth"])
app.include_router(routes_data.router, prefix="/data", tags=["Data"])
app.include_router(routes_alerts.router, prefix="/alerts", tags=["Alerts"])
app.include_router(routes_ml.router, prefix="/ml", tags=["ML"])

@app.get("/")
def root():
    return {"message": "Smart Health Monitoring API is running 🚀"}
''',

    "app/config.py": "pass\n",
    "app/dependencies.py": "pass\n",

    "app/api/__init__.py": "",
    "app/api/routes_auth.py": "from fastapi import APIRouter\n\nrouter = APIRouter()\n",
    "app/api/routes_data.py": "from fastapi import APIRouter\n\nrouter = APIRouter()\n",
    "app/api/routes_alerts.py": "from fastapi import APIRouter\n\nrouter = APIRouter()\n",
    "app/api/routes_ml.py": "from fastapi import APIRouter\n\nrouter = APIRouter()\n",

    "app/core/security.py": "pass\n",
    "app/core/utils.py": "pass\n",

    "app/db/database.py": "pass\n",
    "app/db/models.py": "pass\n",
    "app/db/crud.py": "pass\n",

    "app/ml/model.py": "pass\n",
    "app/ml/predictor.py": "pass\n",

    "app/schemas/auth.py": "pass\n",
    "app/schemas/data.py": "pass\n",
    "app/schemas/ml.py": "pass\n",

    "tests/test_auth.py": "pass\n",
    "tests/test_data.py": "pass\n",
    "tests/test_ml.py": "pass\n",

    "requirements.txt": "fastapi\nuvicorn\nfirebase-admin\nsqlalchemy\npydantic\n",
    "README.md": "# Smart Community Health Monitoring Backend\n"
}

def create_structure():
    # Create directories
    for d in dirs:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)

    # Create files
    for filepath, content in files.items():
        full_path = os.path.join(base_dir, filepath)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

    print(f"✅ Project structure created inside '{base_dir}'")

if __name__ == "__main__":
    create_structure()
