import json
import boto3

s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
s3vectors = boto3.client("s3vectors")
table = dynamodb.Table('s3-vector-chunks-entry')
S3_VECTOR_BUCKET_NAME = 's3-vectors-fraud-document'
S3_VECTOR_INDEX_NAME  = 's3-vector-index-fraud-document'

def lambda_handler(event, context):

    existing_chunk_ids = []
    is_existing_vectors_deleted = False
    
    bucket = event['detail']['bucket']['name']
    key = event['detail']['object']['key']

    print(f"key is {key}")

    existing_chunk_entry = table.get_item(Key={"s3_object_name": key})
    print(f"Existing entry is {existing_chunk_entry}")

    if 'Item' in existing_chunk_entry:
        if 'chunks' in existing_chunk_entry['Item']:
            existing_chunk_ids = existing_chunk_entry['Item']['chunks']

    if existing_chunk_ids:
        delete_response = s3vectors.delete_vectors(
            vectorBucketName=S3_VECTOR_BUCKET_NAME,
            indexName=S3_VECTOR_INDEX_NAME,
            keys=existing_chunk_ids
        )

        if delete_response and delete_response['ResponseMetadata']['HTTPStatusCode'] == 200:
            print(f"Successfully deleted the chunks from S3 Vector database")
            is_existing_vectors_deleted = True
        else:
            print(f"Failed to delete the chunks from S3 Vector database")

    return {
        'statusCode': 200,
        'action': 'delete_existing_chunks',
        'filename': key,
        'is_existing_vectors_deleted': is_existing_vectors_deleted
    }
