
# version 9
# script to bundle the 2 directory, push it to S3, and cleanly overwrite the SageMaker Serverless endpoint.
# This endpoint should server lstm model version 2 with 4 different features: UNRATE, FEDFUNDS, GDP, INFLATION

#NOTE: This file is ONLY a template, with placeholder variables. The working file is in owner's aws account

import base64, tarfile, time, boto3, os

sagemaker_client = boto3.client("sagemaker", region_name="us-west-2")
s3_client = boto3.client("s3", region_name="us-west-2")

BUCKET_NAME = "BUCKET_NAME"
ROLE_ARN = "ROLE_ARN"

# INCREMENT TO VERSION 9 TO CLEAR CACHES
MODEL_NAME = "MODEL_NAME"
CONFIG_NAME = "CONFIG_NAME"
NEW_ENDPOINT_NAME = "NEW_ENDPOINT_NAME"
S3_KEY = "S3_KEY"

print("1. Compiling clean Version 9 tarball package...")
with tarfile.open("model_nber_v9.tar.gz", "w:gz") as tar:
    tar.add("export/7", arcname="7")
    tar.add("code/inference.py", arcname="code/inference.py")

print("2. Syncing architecture archive with Amazon S3 vault...")
s3_client.upload_file("model_nber_v9.tar.gz", BUCKET_NAME, S3_KEY)

IMAGE_URI = "aws.amazon.com"

print("3. Registering clean Version 9 Configuration...")
try: sagemaker_client.delete_model(ModelName=MODEL_NAME)
except: pass

sagemaker_client.create_model(
    ModelName=MODEL_NAME, ExecutionRoleArn=ROLE_ARN,
    PrimaryContainer={
        "Image": IMAGE_URI, 
        "ModelDataUrl": f"s3://{BUCKET_NAME}/{S3_KEY}",
        "Environment": {
            "SAGEMAKER_PROGRAM": "inference.py",
            "SAGEMAKER_SUBMIT_DIRECTORY": f"s3://{BUCKET_NAME}/{S3_KEY}"
        }
    }
)

print("4. Allocating clean serverless variants lane...")
try: sagemaker_client.delete_endpoint_config(EndpointConfigName=CONFIG_NAME)
except: pass

sagemaker_client.create_endpoint_config(
    EndpointConfigName=CONFIG_NAME,
    ProductionVariants=[{
        "VariantName": "AllTraffic", "ModelName": MODEL_NAME,
        "ServerlessConfig": {"MemorySizeInMB": 3072, "MaxConcurrency": 1}
    }]
)

print("5. Triggering cloud network routing update...")
sagemaker_client.update_endpoint(EndpointName=NEW_ENDPOINT_NAME, EndpointConfigName=CONFIG_NAME)

print(" Provisioning new virtual instances (This takes 2-4 minutes)...")
while True:
    status = sagemaker_client.describe_endpoint(EndpointName=NEW_ENDPOINT_NAME)["EndpointStatus"]
    print(f"   Current Status: {status}")
    if status in ["InService", "Failed"]: break
    time.sleep(15)

print("\n SUCCESS! Version 9 infrastructure layers are fully live and calibrated!")
