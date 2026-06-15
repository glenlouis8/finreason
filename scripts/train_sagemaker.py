"""
Launch SFT + DPO training jobs on AWS SageMaker.

Prerequisites:
    pip install sagemaker boto3
    AWS credentials configured (aws configure)
    S3 bucket for data/output

Usage:
    python scripts/train_sagemaker.py --stage sft --bucket my-bucket --role arn:aws:iam::...
    python scripts/train_sagemaker.py --stage dpo --bucket my-bucket --role arn:aws:iam::...
"""

import argparse
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace


INSTANCE_TYPE = "ml.g5.2xlarge"  # 24GB A10G — cheapest single-GPU for 7B QLoRA
TRANSFORMERS_VERSION = "4.36"
PYTORCH_VERSION = "2.1"
PY_VERSION = "py310"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["sft", "dpo"], required=True)
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--role", required=True, help="SageMaker IAM role ARN")
    parser.add_argument("--instance", default=INSTANCE_TYPE)
    parser.add_argument("--region", default="us-east-1")
    return parser.parse_args()


def upload_data(bucket: str, region: str) -> dict:
    session = sagemaker.Session(boto_session=boto3.Session(region_name=region))
    s3_data = {}
    for split in ["sft_train", "sft_eval", "dpo_pairs"]:
        local_path = f"data/{split}.jsonl"
        s3_uri = session.upload_data(
            path=local_path,
            bucket=bucket,
            key_prefix=f"finreason/data",
        )
        s3_data[split] = s3_uri
        print(f"Uploaded {local_path} → {s3_uri}")
    return s3_data


def launch_sft(args, session: sagemaker.Session):
    hyperparameters = {
        "config": "configs/sft.yaml",
    }

    estimator = HuggingFace(
        entry_point="train_sft.py",
        source_dir=".",
        role=args.role,
        instance_type=args.instance,
        instance_count=1,
        transformers_version=TRANSFORMERS_VERSION,
        pytorch_version=PYTORCH_VERSION,
        py_version=PY_VERSION,
        hyperparameters=hyperparameters,
        environment={
            "WANDB_PROJECT": "finreason",
            "HF_TOKEN": "",  # set via SageMaker env or Secrets Manager
        },
        output_path=f"s3://{args.bucket}/finreason/checkpoints/sft",
        sagemaker_session=session,
    )

    print("Launching SFT training job...")
    estimator.fit(
        inputs={"data": f"s3://{args.bucket}/finreason/data"},
        job_name="finreason-sft",
        wait=False,
    )
    print(f"Job launched: finreason-sft")
    print(f"Monitor: https://console.aws.amazon.com/sagemaker/home#/jobs")
    return estimator


def launch_dpo(args, session: sagemaker.Session):
    hyperparameters = {
        "config": "configs/dpo.yaml",
        "sft_config": "configs/sft.yaml",
    }

    estimator = HuggingFace(
        entry_point="train_dpo.py",
        source_dir=".",
        role=args.role,
        instance_type=args.instance,
        instance_count=1,
        transformers_version=TRANSFORMERS_VERSION,
        pytorch_version=PYTORCH_VERSION,
        py_version=PY_VERSION,
        hyperparameters=hyperparameters,
        environment={
            "WANDB_PROJECT": "finreason",
            "HF_TOKEN": "",
        },
        output_path=f"s3://{args.bucket}/finreason/checkpoints/dpo",
        sagemaker_session=session,
    )

    print("Launching DPO training job...")
    estimator.fit(
        inputs={
            "data": f"s3://{args.bucket}/finreason/data",
            "sft_checkpoint": f"s3://{args.bucket}/finreason/checkpoints/sft",
        },
        job_name="finreason-dpo",
        wait=False,
    )
    print(f"Job launched: finreason-dpo")
    return estimator


def main():
    args = parse_args()
    session = sagemaker.Session(
        boto_session=boto3.Session(region_name=args.region)
    )

    if args.stage == "sft":
        upload_data(args.bucket, args.region)
        launch_sft(args, session)
    elif args.stage == "dpo":
        launch_dpo(args, session)


if __name__ == "__main__":
    main()
