import json
import urllib.parse
import boto3

comprehend = boto3.client("comprehend")

print('Loading function')

s3 = boto3.client('s3')


def lambda_handler(event, context):
    #print("Received event: " + json.dumps(event, indent=2))

    # Get the object from the event and show its content type
    #print(f"Full payload to lambda is {event}")
    #bucket = event['Records'][0]['s3']['bucket']['name']
    #key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')

    bucket = event['detail']['bucket']['name']
    key = event['detail']['object']['key']
    pii_summary = {}

    print(f"Reading file {key} from bucket {bucket}")

    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        print(f"Read document content of length {len(content)}")
        
        print("Using Comprehend API to detect PII entities")

        comprehendPIIEntities = comprehend.detect_pii_entities(
            Text=content,
            LanguageCode="en")

        print(f"Comprehend detected PII entities are {comprehendPIIEntities}")

        if not comprehendPIIEntities:
            print("No PII entities identified. Passing original document")
        else:
            print("Identified PII entities. Redacting the document")

        for entity in reversed(comprehendPIIEntities['Entities']):
            entity_type = entity['Type']

            if entity_type in pii_summary:
                pii_summary[entity_type] += 1
            else:
                pii_summary[entity_type] = 1

            maskedString = '['+ entity['Type'] + ']'
            content = content[: entity['BeginOffset']] + maskedString + content[entity['EndOffset']: ]

        return {
            "content": content,
            "pii_summary": pii_summary
        }

    except Exception as e:
        print(e)
        print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(key, bucket))
        raise e
