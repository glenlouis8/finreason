"""
Deploy FinReason DPO model to AWS SageMaker endpoint.

Uses the published HuggingFace model directly — no S3 upload needed.

Prerequisites:
    pip install sagemaker boto3
    AWS credentials configured (aws configure)

Usage:
    # Deploy endpoint
    python scripts/deploy_sagemaker.py --role arn:aws:iam::... --deploy

    # Run inference against live endpoint
    python scripts/deploy_sagemaker.py --endpoint finreason-dpo --predict

    # Delete endpoint (stop billing)
    python scripts/deploy_sagemaker.py --endpoint finreason-dpo --delete
"""

import argparse
import json
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel


HF_MODEL_ID = "glen-louis/finreason-qwen2.5-7b-dpo"
ENDPOINT_NAME = "finreason-dpo"
INSTANCE_TYPE = "ml.g5.2xlarge"  # 24GB A10G


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", help="SageMaker IAM role ARN")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--endpoint", default=ENDPOINT_NAME)
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--delete", action="store_true")
    return parser.parse_args()


def deploy(args, session: sagemaker.Session):
    hub_config = {
        "HF_MODEL_ID": HF_MODEL_ID,
        "HF_TASK": "text-generation",
        "SM_NUM_GPUS": "1",
    }

    model = HuggingFaceModel(
        env=hub_config,
        role=args.role,
        transformers_version="4.36",
        pytorch_version="2.1",
        py_version="py310",
        sagemaker_session=session,
    )

    print(f"Deploying {HF_MODEL_ID} to {args.endpoint}...")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=args.endpoint,
    )
    print(f"Endpoint live: {args.endpoint}")
    print(f"Monitor: https://console.aws.amazon.com/sagemaker/home#/endpoints/{args.endpoint}")
    return predictor


def predict(args, session: sagemaker.Session):
    from sagemaker.huggingface import HuggingFacePredictor

    predictor = HuggingFacePredictor(
        endpoint_name=args.endpoint,
        sagemaker_session=session,
    )

    sample_input = {
        "inputs": (
            "Context:\nNet revenue increased from $10.2 billion in 2021 to $12.5 billion in 2022.\n\n"
            "Table:\nYear | Revenue\n2021 | 10.2\n2022 | 12.5\n\n"
            "Question: What was the percentage increase in revenue from 2021 to 2022?"
        ),
        "parameters": {
            "max_new_tokens": 256,
            "do_sample": False,
        },
    }

    print("Sending request...")
    response = predictor.predict(sample_input)
    print("Response:")
    print(json.dumps(response, indent=2))


def delete_endpoint(args, session: sagemaker.Session):
    client = session.boto_session.client("sagemaker")
    client.delete_endpoint(EndpointName=args.endpoint)
    print(f"Endpoint {args.endpoint} deleted.")


def main():
    args = parse_args()
    session = sagemaker.Session(
        boto_session=boto3.Session(region_name=args.region)
    )

    if args.deploy:
        deploy(args, session)
    elif args.predict:
        predict(args, session)
    elif args.delete:
        delete_endpoint(args, session)
    else:
        print("Pass --deploy, --predict, or --delete")


if __name__ == "__main__":
    main()
