import json

def lambda_handler(event, context):
    # TODO implement

    all_chunk_response = []
    chunk_ids = []

    for chunk_result in event:
        metadata = {}
        embedding_result = {}
        for result in chunk_result:
            if result['action'] == 'get_metadata':
                metadata = result['metadata']
            elif result['action'] == 'generate_embeddings':
                embedding_result = result['embedding_result']
                chunk_ids.append(embedding_result['id'])

        embedding_result['metadata'] = metadata
        all_chunk_response.append(embedding_result)

    return {
        'statusCode': 200,
        'action': 'merge_chunk_embedding_metadata',
        'result': all_chunk_response,
        'all_chunk_ids': chunk_ids
    }
