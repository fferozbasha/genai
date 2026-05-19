import json
import boto3

s3vectors = boto3.client("s3vectors")

S3_VECTOR_BUCKET_NAME = 's3-vectors-fraud-document'
S3_VECTOR_INDEX_NAME  = 's3-vector-index-fraud-document'

def lambda_handler(event, context):
    # TODO implement
    """
    {
    "indexArn": "string",
    "indexName": "string",
    "vectorBucketName": "string",
    "vectors": [ 
        { 
            "data": { ... },
            "key": "string",
            "metadata": JSON value
        }
    ]
    }
    This is the format to put the vectors. 
    Have to get the vectors from the lambda input and pass to the 
    put_vectors api
    """

    results = event['result']

    if not results:
        return {
            'statusCode': 500,
            'body': json.dumps('No results')
        }

    vectors = []

    for event in results:

        metadata = event["metadata"]
        print(f"Metadata: {metadata}")
        id = event['id']
        data = event['embedding']
        text = event ['text']
        vector = {
            "key": id,
            "data": {'float32': data},
            "metadata": {
                "text": text,
                **metadata
            }
        }

        vectors.append(vector)

    s3_put_vectors_response = s3vectors.put_vectors(
        indexName = S3_VECTOR_INDEX_NAME,
        vectorBucketName= S3_VECTOR_BUCKET_NAME,
        #indexArn= 'arn:aws:s3vectors:ap-southeast-2:414079114528:bucket/s3-vectors-fraud-document/index/s3-vector-index-fraud-document',
        vectors= vectors
    )

    print("S3 Put Vector completed")
    print(s3_put_vectors_response)

    return {
        'statusCode': 200,
        'total_vectors_inserted': len(vectors)
    }
