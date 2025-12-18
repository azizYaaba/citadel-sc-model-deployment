import mlflow
import os
from pathlib import Path

def setup_mlflow():
    """
    Sets up MLflow tracking and registry URIs to point to a shared sqlite database
    located at the root of the project.
    """
    # Get the root directory of the project (assuming this file is in shared/)
    root_dir = Path(__file__).resolve().parents[1]
    db_path = root_dir / "mlflow.db"
    
    tracking_uri = f"sqlite:///{db_path}"
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    
    print(f"✅ MLflow configured with URI: {tracking_uri}")
