import json
import boto3
from botocore.config import Config

bedrock_instance = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",
    config=Config(read_timeout=3600)
    )

def lambda_handler(event, context):

    chunk = event['chunk']
    prompt = f"""
    You are an expert document classification and metadata extraction system.

    Analyze the below document chunk and extract meaningful metadata.

    Return ONLY valid JSON.

    Extract:
    - topic
    - subtopic
    - department
    - risk_category
    - document_type
    - sensitivity
    - keywords (array of important keywords)

    Document Chunk:
    {chunk}
    """

    llm_response = bedrock_instance.converse(
        modelId="us.amazon.nova-2-lite-v1:0",
        messages=[
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ]
    )
    response_text = llm_response["output"]["message"]["content"][0]["text"]
    cleaned_json = (response_text.replace("```json", "").replace("```", "").strip())

    # TODO implement
    return {
        'statusCode': 200,
        'action': 'get_metadata',
        'metadata': json.loads(cleaned_json)
    }
