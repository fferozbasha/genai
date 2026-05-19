import json
import boto3

dynamodb = boto3.resource('dynamodb')
s3vectors = boto3.client("s3vectors")
table = dynamodb.Table('s3-vector-chunks-entry')
S3_VECTOR_BUCKET_NAME = 's3-vectors-fraud-document'
S3_VECTOR_INDEX_NAME  = 's3-vector-index-fraud-document'


def lambda_handler(event, context):
    # TODO implement

    existing_chunk_ids = []
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    print(f"key is {key}")

    existing_chunk_entry = table.get_item(Key={"s3_object_name": key})
    if 'Item' in existing_chunk_entry:
        if 'chunks' in existing_chunk_entry['Item']:
            existing_chunk_ids = existing_chunk_entry['Item']['chunks']

    if existing_chunk_ids:
        delete_response = s3vectors.delete_vectors(
            vectorBucketName=S3_VECTOR_BUCKET_NAME,
            indexName=S3_VECTOR_INDEX_NAME,
            keys=existing_chunk_ids
        )

        if (delete_response["ResponseMetadata"]["HTTPStatusCode"]== 200):
            table.delete_item(Key={"s3_object_name": key})


    return {
        'statusCode': 200,
        'deleted_vectors': len(existing_chunk_ids),
        "file_name": key
    }
