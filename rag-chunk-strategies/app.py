import boto3
import json
import sys
import re
from botocore.config import Config
import chromadb

from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = "data/content.txt"

bedrock_instance = None
rag_document = None
vector_db_client = None

user_options = [
    {"option_name": "100Split", "chunk_size": 100, "chunk_overlap": 10},
    {"option_name": "300Split", "chunk_size": 300, "chunk_overlap": 50},
    {"option_name": "600Split", "chunk_size": 600, "chunk_overlap": 100},
    {"option_name": "1000Split", "chunk_size": 1000, "chunk_overlap": 150}
]

def load_document(file_path="data/content.txt"):
    global rag_document
    with open(file_path, "r", encoding="utf-8") as f:
        rag_document = f.read()
    
    print("RAG Document loaded successfully.")

def split_chunks(chunk_size=None, chunk_overlap=None):
    """
    Splits the RAG document in to multiple chunks based on the chunk size and overlap
    """

    if not chunk_size or not chunk_overlap:
        raise("Invalid option for chunk size and chunk overlap. Exiting...")

    if not rag_document:
        raise("No document provided to split the chunks. Exiting...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_text(rag_document)
    return chunks

def init_bedrock_instance():
    
    global bedrock_instance 

    bedrock_instance = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",
    config=Config(read_timeout=3600)
    )

    print("Bedrock instance initiated successfully.")

def get_llm_response(prompt):
    llm_response = bedrock_instance.converse(
        modelId="us.amazon.nova-2-lite-v1:0",
        messages=[
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ]
    )

    return llm_response

def get_text_embedding(text=None):
    if not text:
        print("No text provided to generate embedding. Exiting ....")
        sys.exit()

    request = json.dumps({"inputText": text})

    embedding_model_response = bedrock_instance.invoke_model(modelId='amazon.titan-embed-text-v2:0', body=request)

    embedding_model_response_output = json.loads(embedding_model_response['body'].read())['embedding']

    return embedding_model_response_output

def get_all_chunks_embeddings(chunks=None):
    """
    Iterates through all the chunks (array) and sends request to the 
    embedding model to get the embeddings for the chunk text
    """

    if not chunks:
        print(f"No chunks available to get embeddings.")
        sys.exit()

    print("Embedding generation for Chunks. Started....")

    all_chunks_embeddings = []

    for chunk in chunks:
        chunk_embedding_response = get_text_embedding(chunk)
        all_chunks_embeddings.append(chunk_embedding_response)
    
    print("Embedding generation for Chunks. Completed....")

    return all_chunks_embeddings

def init_vector_db_client():
    global vector_db_client
    vector_db_client =  chromadb.Client()
    print("Vector DB Client created successfully.")

def init_vector_db_instance(collection_name=None):

    if not collection_name:
        print("Please provide the collection name to be created in ChromaDB. Exiting....")
        sys.exit()

    collection = vector_db_client.create_collection(collection_name)
    
    print(f"Vector DB collection created with name = {collection_name}")

    return collection

def get_prompt_with_rag_results(user_query, rag_results):
    """
    Embeds the User input query and the RAG results (context) in a prompt to be fed in to 
    LLM to get a consolidated response. 
    """

    context = ""

    for rag_result in rag_results['documents'][0]:
        context += rag_result
        context += "\n"

    prompt = f"""
    Please provide a short response in 2-3 lines max.
    Question: {user_query}
    Context: {context}
    """ 

    return prompt

def evaluate_rag_response(query, llm_response, expected_response):
    """
    Uses the LLM as judge. Passes the Query, the LLM response and the expected response.
    Asks teh LLM to evaluate for all the key aspects of the RAG performance evaluation. 
    """

    prompt = f"""
You are an expert evaluator.

Evaluate the generated answer based on the following criteria:

1. Correctness (1-5)
2. Completeness (1-5)
3. Relevance (1-5)
4. Groundedness (1-5) - Is the answer supported by the context?
5. Clarity (1-5)

Return ONLY valid JSON in this format:

{{
  "correctness": int,
  "completeness": int,
  "relevance": int,
  "groundedness": int,
  "clarity": int,
  "overall_score": int,
  "reason": "short explanation"
}}

Question: {query}
Expected Answer: {expected_response}
Generated Answer: {llm_response}
"""

    return get_llm_response(prompt=prompt)

def extract_json_llm_response(response):
    # Remove ```json or ``` wrappers
    response = re.sub(r"```json|```", "", response).strip()
    
    # Extract JSON block
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        return json.loads(match.group())
    
    return None

load_document()
init_bedrock_instance()
init_vector_db_client()

print("\n")

user_query_input = input("User: ")
user_input_expected_response = input("Expected Response: ")
user_query_input_embedding = get_text_embedding(user_query_input)

all_evaluation_results = []

for user_option in user_options:

    user_option_name=user_option['option_name']
    user_option_chunk_size = user_option['chunk_size']
    user_option_chunk_overlap = user_option['chunk_overlap']

    print("="*100)
    print(f"Starting with User option {user_option_name} with Chunk Size = {user_option_chunk_size} and Chunk overlap = {user_option_chunk_overlap}")
    print("\n")

    chunks = split_chunks(chunk_size=user_option["chunk_size"], chunk_overlap=user_option["chunk_overlap"])

    print(f"Document split in to {len(chunks)} chunks.")

    chunks_embeddings = get_all_chunks_embeddings(chunks=chunks)
    chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]

    vector_db_instance = init_vector_db_instance(collection_name=user_option["option_name"])

    print("Insert the RAG document chunk embeddings in to the Vector DB collection. Starting...")
    vector_db_instance.add(embeddings=chunks_embeddings, ids=chunk_ids, documents=chunks)
    print("Insert the RAG document chunk embeddings in to the Vector DB collection. Completed.")

    print("Querying the collection now with the User query and retrieving top 3 results")
    rag_results = vector_db_instance.query(query_embeddings=user_query_input_embedding, n_results=3)

    prompt = get_prompt_with_rag_results(user_query_input, rag_results=rag_results)

    llm_response = get_llm_response(prompt)

    print(f"\nLLM Response: {llm_response["output"]["message"]["content"][0]["text"]}")

    evaluation_results = evaluate_rag_response(
        user_query_input, 
        llm_response=llm_response, 
        expected_response=user_input_expected_response)["output"]["message"]["content"][0]["text"]
    
    evaluation_results = extract_json_llm_response(evaluation_results)
    evaluation_results['option'] = user_option_name
    evaluation_results['chunk_size'] = user_option_chunk_size
    evaluation_results['chunk_overlap'] = user_option_chunk_overlap

    all_evaluation_results.append(evaluation_results)
    
print("\n")
print("-"*150)

for result in all_evaluation_results:
    print(f"Option        = {result['option']}, Chunk Size={result['chunk_size']}, Overlap={result['chunk_overlap']}")
    print(f"Correctness   = {result['correctness']}")
    print(f"Completeness  = {result['completeness']}")
    print(f"Relevance     = {result['relevance']}")
    print(f"Groundedness  = {result['groundedness']}")
    print(f"Clarity       = {result['clarity']}")
    print(f"Overall_score = {result['overall_score']}")
    print(f"Reason.       = {result['reason']}")
    print("\n")

