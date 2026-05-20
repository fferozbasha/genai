# Event-Driven RAG Pipeline with Amazon S3 Vectors

## Overview

This project demonstrates an end-to-end event-driven Retrieval-Augmented Generation (RAG) pipeline built using AWS services including:

- Amazon S3
- Amazon S3 Vectors
- Amazon Bedrock
- AWS Lambda
- AWS Step Functions
- Amazon EventBridge
- Amazon DynamoDB
- Amazon Comprehend
- Streamlit

The solution automatically ingests documents uploaded into Amazon S3, generates embeddings and metadata, stores vectors in Amazon S3 Vectors, and provides a Streamlit-based chat interface for semantic retrieval.

---

## Key Features

### Automated RAG Ingestion Pipeline

Whenever a file is uploaded into the source S3 bucket:

- EventBridge detects the S3 object creation event
- AWS Step Functions orchestrates the ingestion workflow
- Lambda functions perform chunking, metadata generation, and embedding generation
- Vectors are inserted into Amazon S3 Vectors

---

### PII Detection and Redaction

Amazon Comprehend is used to:

- Detect PII entities
- Redact sensitive information
- Generate PII summaries

---

### Vector Lifecycle Management

To ensure synchronization between source documents and the vector database:

- Chunk IDs are tracked using DynamoDB
- Existing vectors are automatically deleted when files are updated
- Vectors are automatically removed when source files are deleted

---

### Semantic Chat Interface

A Streamlit frontend provides:

- Document upload interface
- Semantic chat interface
- RAG pipeline monitoring
- File audit dashboard

---

## Architecture

### Ingestion Workflow

1. User uploads document into Amazon S3
2. EventBridge detects object creation event
3. Step Functions workflow starts
4. Document is chunked
5. PII entities are detected and redacted
6. Metadata is generated using Amazon Bedrock
7. Embeddings are generated using Bedrock embedding models
8. Vectors are inserted into Amazon S3 Vectors
9. Audit information is stored in DynamoDB

---

### Cleanup Workflow

#### Scenario 1: File Updated

- Existing chunk IDs are retrieved from DynamoDB
- Old vectors are deleted from Amazon S3 Vectors
- Fresh vectors are generated and inserted

#### Scenario 2: File Deleted

- Lambda function is triggered by S3 delete event
- Associated chunk IDs are retrieved from DynamoDB
- Corresponding vectors are deleted from Amazon S3 Vectors
- Audit records are removed from DynamoDB

---

## Streamlit Frontend

### Upload Document

- Upload documents directly into Amazon S3
- Track Step Function execution status
- View ingestion summary and PII statistics

### Chat

- Ask questions against uploaded documents
- Retrieve relevant vectors from Amazon S3 Vectors
- Generate grounded responses using Amazon Bedrock

### List Files

- View processed files
- View vector counts
- View PII summaries
- View ingestion audit details

---

## Technologies Used

| Service | Purpose |
|---|---|
| Amazon S3 | Source document storage |
| Amazon S3 Vectors | Vector database |
| Amazon Bedrock | Embeddings and LLM |
| AWS Lambda | Processing functions |
| AWS Step Functions | Workflow orchestration |
| Amazon EventBridge | Event-driven triggering |
| Amazon DynamoDB | Audit and lifecycle tracking |
| Amazon Comprehend | PII detection |
| Streamlit | Frontend UI |

---

## Key Learnings

This project provided hands-on exposure to:

- Event-driven architectures
- Serverless orchestration
- RAG systems
- Vector databases
- Semantic retrieval
- Lifecycle management
- Amazon S3 Vectors
- Amazon Bedrock
- Streamlit frontend development

One of the most valuable learnings was understanding how important synchronization and cleanup workflows are in real-world Gen AI systems.

---

## Author

Feroz Basha

---