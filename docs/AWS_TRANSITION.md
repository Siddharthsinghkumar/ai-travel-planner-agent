# AWS Transition Guide (Infrastructure & Storage)

Currently, the L'Évasion travel agent uses **Postgres** (via `langgraph-checkpoint-postgres`) for LangGraph state checkpointing and **Cloudflare R2** for S3-compatible blob storage. This stack is designed to be fully open-source and free/low-cost.

If the scale requires migrating to AWS native services, follow this guide to replace Postgres and Cloudflare R2 with **AWS DynamoDB** and **AWS S3**.

## 1. State Checkpointing: Postgres -> DynamoDB
To migrate the LangGraph checkpointing from Postgres to DynamoDB:

1. Update `requirements.txt`:
   Remove `langgraph-checkpoint-postgres`.
   Add `langgraph-checkpoint-dynamodb` (or equivalent boto3-based Saver if using custom implementation).
   
2. Update `core/storage.py`:
   Replace `PostgresSaver` with `DynamoDBSaver`.
   ```python
   # Example implementation with DynamoDB
   from langgraph.checkpoint.dynamodb import DynamoDBSaver
   import boto3

   def get_checkpointer():
       client = boto3.client("dynamodb", region_name="us-east-1")
       return DynamoDBSaver(client=client, table_name="langgraph_checkpoints")
   ```
   
3. Provision DynamoDB Table:
   Ensure the target table uses `thread_id` (or similar) as the partition key to store the state correctly.

## 2. Blob Storage: Cloudflare R2 -> AWS S3
Since Cloudflare R2 is S3-compatible, transitioning to AWS S3 primarily involves configuration changes rather than code rewriting.

1. Remove Cloudflare R2 specific endpoint URLs in your S3 client configuration.
2. Update environment variables to point to the AWS region and standard S3 endpoint:
   ```env
   AWS_ACCESS_KEY_ID=<aws_access_key>
   AWS_SECRET_ACCESS_KEY=<aws_secret_key>
   AWS_REGION=us-east-1
   S3_BUCKET_NAME=your-aws-s3-bucket
   ```
3. Ensure IAM policies on AWS allow the necessary `s3:PutObject`, `s3:GetObject`, and `s3:ListBucket` permissions.
