import boto3
import json
import sys
from botocore.config import Config
import numpy as np

# Creating the bedrock client instance
bedrock = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",
    config=Config(read_timeout=3600),
)

# List of documents to mimic something similar to multiple RAG documents
documents = [
    "NovaTech Industries was founded in 2015 in San Francisco by Elena Verma.",
    "The company specializes in artificial intelligence and cloud computing solutions.",
    "NovaTech reported a revenue of 120 million dollars in 2022.",
    "By 2024, NovaTech's annual revenue grew to 350 million dollars.",
    "NovaTech employs over 1,200 people across 5 countries.",
    "The company launched its flagship AI platform NovaMind in 2018.",
    "NovaMind helps businesses automate customer support using large language models.",
    "NovaTech expanded into Europe in 2019 with its London office.",
    "In 2021, NovaTech secured Series C funding of 80 million dollars.",
    "The company’s cloud infrastructure is built on a multi-region architecture.",
    "NovaTech uses vector databases to power semantic search capabilities.",
    "Their AI models are fine-tuned for industries like healthcare and finance.",
    "NovaTech offers a subscription-based pricing model for its enterprise clients.",
    "In 2023, NovaTech introduced real-time analytics dashboards for businesses.",
    "The company partners with AWS and Azure for cloud services.",
    "NovaTech has over 3,000 enterprise customers worldwide.",
    "Security and data privacy are key priorities, with ISO 27001 certification.",
    "NovaTech’s R&D team focuses on generative AI and autonomous agents.",
    "The company plans to go public with an IPO by 2027.",
    "NovaTech aims to reach 1 billion dollars in revenue by 2030."
]

# List placeholder to keep track of all the RAG document embedding details
rag_document_embeddings = []

# Function to retreive embedding for the document passed using the Embedding model
def get_documents_embeddings(docs=None):
    if not docs:
        print("No document provided to generate embedding. Exiting ....")
        exit

    request = json.dumps({"inputText": docs})

    embedding_model_response = bedrock.invoke_model(modelId='amazon.titan-embed-text-v2:0', body=request)
    return embedding_model_response

# Calculating the cosine similarity using numpy
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Calculating the embedding for each of the RAG document
# Storing the embedding value alongside the actual document text
# Appending everything to array rag_document_embeddings

print(f"\nRetreiving embeddings for the {len(documents)} RAG documents ...")

for doc in documents:
    embedding_stream = get_documents_embeddings(doc)
    embedding = json.loads(embedding_stream['body'].read())
    rag_doc = {}
    rag_doc["text"] = doc
    rag_doc["embedding"] = embedding["embedding"]
    rag_document_embeddings.append(rag_doc)

# Getting the user input
user_input = input("\nYou: ")

if user_input == 'exit':
    print("Exiting ....")
    sys.exit()

user_input_embedding_stream = get_documents_embeddings(user_input)
user_input_embedding_value = json.loads(user_input_embedding_stream['body'].read())

print("\nCalculating the cosine similarity score comparing the embedding of RAG document"
"and the user input")
for rag_doc in rag_document_embeddings:
    similarity_score = cosine_similarity(rag_doc["embedding"], user_input_embedding_value['embedding'])
    
    rag_doc["score"] = similarity_score

# filtering the RAG documents only which has similarity score more than 0.5
# this is to ensure that if the user input is completely irrelevant to RAG documents,
# then nothing will be considered. 
filtered_rag_docs = [doc for doc in rag_document_embeddings if doc["score"] > 0.5]

if len(filtered_rag_docs) < 2:
    print(f"\nYour query {user_input} does not seem to be relevant to any of the RAG documents. Please query correctly")
    print("Exiting ....")
    sys.exit()

# sorting based on the similarity score
filtered_rag_docs.sort(key=lambda x: x["score"], reverse=True)

# picking top 3 rag documents
print("\nSelecting top 3 related documents based on the user input")
top_rag_docs = filtered_rag_docs[:3]

# Augmenting the prompt along with RAG output (top 3 related documents)
rag_augmented_prompt = user_input
rag_augmented_prompt += "\nPlease provide a short response in 2-3 lines max."
rag_augmented_prompt += '\nConsider the following and answer: \n'

for top_rag in top_rag_docs:
    print(f"RAG doc = {top_rag['text']}, similarity_score = {top_rag['score']:.4f}")
    rag_augmented_prompt += top_rag['text']

print("\nAugmented RAG Query")
print("========================")
print(rag_augmented_prompt)

# finally, invoking the FM model with the augmented query to get the response
# based on user input and as well the related RAG documents
fm_model_response = bedrock.converse(
        modelId="us.amazon.nova-2-lite-v1:0",
        messages=[
            {
                "role": "user",
                "content": [{"text": rag_augmented_prompt}]
            }
        ]
    )

reply = fm_model_response["output"]["message"]["content"][0]["text"]
print("\nRAG Based AI Response")
print("==========================")
print(reply)