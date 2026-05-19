import json
import boto3
from botocore.config import Config
import uuid

# Creating the bedrock client instance
bedrock_instance = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",
    config=Config(read_timeout=3600),
)

def lambda_handler(event, context):
    # TODO implement

    chunk = event['chunk']
    chunk_id = event['index']

    request = json.dumps({"inputText": chunk})
    embedding_model_response = bedrock_instance.invoke_model(modelId='amazon.titan-embed-text-v2:0', body=request)
    embedding_model_response_value = json.loads(embedding_model_response['body'].read())
    embedding = embedding_model_response_value["embedding"]
    response_object = {
        'id': str(uuid.uuid4()), 
        'text': chunk,
        'embedding': embedding
    }

    return {
        'statusCode': 200,
        'action': 'generate_embeddings',
        'embedding_result': response_object
    }