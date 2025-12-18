from __future__ import annotations
import argparse
from mlflow.tracking import MlflowClient
import sys
from pathlib import Path

# Add root directory to sys.path to allow importing shared modules
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from shared.mlflow_config import setup_mlflow
MODEL_NAME="IrisClassifier"
def main():
    setup_mlflow()
    p=argparse.ArgumentParser()
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--stage", choices=["Staging","Production","Archived"], required=True)
    args=p.parse_args()
    MlflowClient().transition_model_version_stage(
        name=MODEL_NAME, version=str(args.version), stage=args.stage,
        archive_existing_versions=(args.stage=="Production")
    )
    print(f"✅ {MODEL_NAME} v{args.version} -> {args.stage}")
if __name__=="__main__":
    main()
