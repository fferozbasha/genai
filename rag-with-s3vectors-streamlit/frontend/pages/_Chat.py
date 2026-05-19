import streamlit as st
import boto3
from botocore.config import Config
import json

st.title("Chat")
s3vectors = boto3.client("s3vectors")

bedrock_instance = None
BEDROCK_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
BEDROCK_LLM_MODEL_ID = "us.amazon.nova-2-lite-v1:0"

S3_VECTOR_BUCKET_NAME = 's3-vectors-fraud-document'
S3_VECTOR_INDEX_NAME  = 's3-vector-index-fraud-document'


def _init_bedrock_instance():
    global bedrock_instance
    bedrock_instance = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",
    config=Config(read_timeout=3600),
    )


def get_text_embedding(text):
    request = json.dumps({"inputText": text})
    embedding_model_response = bedrock_instance.invoke_model(modelId=BEDROCK_EMBEDDING_MODEL_ID, body=request)
    embedding_model_response_value = json.loads(embedding_model_response['body'].read())
    embedding = embedding_model_response_value["embedding"]
    return embedding

def query_vector_db(prompt, topK=5):
    if not prompt:
        return "None"
    
    user_input_embedding = get_text_embedding(prompt)
    if not user_input_embedding:
        return "None"
    
    
    response = s3vectors.query_vectors(
        vectorBucketName=S3_VECTOR_BUCKET_NAME,
        indexName=S3_VECTOR_INDEX_NAME,
        queryVector={"float32": user_input_embedding},
        topK=topK,
        returnMetadata = True, 
        returnDistance = True
        )

    return response

def get_relevant_texts_from_vector_db(prompt):

    relevant_texts = []

    query_vectors_response = query_vector_db(prompt=prompt)

    if not query_vectors_response:
        print(f"No response retreived from Vector DB")

    vectors = query_vectors_response.get("vectors", None)
    for vector in vectors:
        print(f"Key = {vector["key"]} with distance={vector["distance"]}")   
        relevant_texts.append(vector["metadata"]["text"])

    return "\n".join(relevant_texts)

def get_prompt_with_rag_results_for_user_query(prompt):

    relevant_texts = get_relevant_texts_from_vector_db(prompt)

    return f"""
        Answer the question using ONLY the below context.

        Context:
        {relevant_texts}

        Question:
        {prompt}
        """

def query_llm_with_user_prompt_rag_texts(prompt):

    updated_prompt = get_prompt_with_rag_results_for_user_query(prompt=prompt)

    llm_response = bedrock_instance.converse(
    modelId=BEDROCK_LLM_MODEL_ID,
    messages=[
            {
                "role": "user",
                "content": [{"text": updated_prompt}]
            }
        ]
    )

    prompt_response = llm_response["output"]["message"]["content"][0]["text"]

    return prompt_response


if not bedrock_instance:
    _init_bedrock_instance()

if "messages" not in st.session_state:
    st.session_state.messages = []

# show history
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])

# user input
prompt = st.chat_input("Ask something...")

if prompt:

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner("Searching knowledge base and generating response..."):
        response = query_llm_with_user_prompt_rag_texts(prompt)
    print(f"Assistant response is {response}")    

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    with st.chat_message("assistant"):
        st.write(response)