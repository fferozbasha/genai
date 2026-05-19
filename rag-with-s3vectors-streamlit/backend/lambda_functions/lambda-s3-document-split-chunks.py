import json

def chunk_text(text, chunk_size=500, overlap=50):

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

def lambda_handler(event, context):
    # TODO implement
    print(f"event in new lambda is {event['content']}")
    chunks = chunk_text(event['content'])
    print(f"Split chunks are {chunks}")

    return {
        'chunks': chunks
    }
