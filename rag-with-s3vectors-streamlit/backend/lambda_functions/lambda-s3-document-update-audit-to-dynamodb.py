import json
import boto3
from datetime import datetime

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('s3-vector-chunks-entry')

def lambda_handler(event, context):
    # TODO implement

    file_name = event["file_name"]
    pii_summary = event["pii_summary"]
    total_vectors_inserted = event["total_vectors_inserted"]
    execution_arn = event["execution_arn"]
    is_existing_vectors_deleted = event["is_existing_vectors_deleted"]
    all_chunk_ids = event["all_chunk_ids"]

    update_audit_response = table.put_item(
        Item={
                's3_object_name': file_name,
                'pii_summary': pii_summary,
                'total_vectors_inserted': total_vectors_inserted,
                'executionArn': execution_arn,
                'timestamp': datetime.utcnow().isoformat(),
                'is_existing_vectors_deleted': is_existing_vectors_deleted,
                'chunks': all_chunk_ids
            }
        )

    print(f"update_audit_response = {update_audit_response}")
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
