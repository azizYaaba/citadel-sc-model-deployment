from __future__ import annotations
import argparse
from mlflow.tracking import MlflowClient
MODEL_NAME="IrisClassifier"
def main():
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
