import streamlit as st

st.title("Enterprise RAG Pipeline with Amazon S3 Vectors")

st.markdown("""
This project demonstrates an event-driven,
serverless Retrieval-Augmented Generation (RAG)
pipeline using AWS services.

### Features
- Automated document ingestion
- PII detection and redaction
- Parallel chunk processing
- Vector embeddings with Amazon Bedrock
- Semantic search using S3 Vectors
- Streamlit chatbot interface
- Lifecycle cleanup for deleted documents

### AWS Services Used
- AWS Lambda
- AWS Step Functions
- Amazon Bedrock
- Amazon S3 Vectors
- DynamoDB
- Streamlit

Use the sidebar to:
- Upload documents
- Chat with the knowledge base
""")