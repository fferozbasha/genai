# RAG Chunking Strategy Evaluation

## 🚀 Overview

This project explores how different chunking strategies impact the
performance of Retrieval-Augmented Generation (RAG) systems.

Instead of just implementing RAG, this project focuses on **evaluation
and experimentation**, which is critical in real-world GenAI systems.

------------------------------------------------------------------------

## 🧠 Problem Statement

How do chunk size and overlap affect: - Retrieval accuracy - Answer
quality

------------------------------------------------------------------------

## ⚙️ Tech Stack

-   Python
-   AWS Bedrock (Titan Embeddings + Nova LLM)
-   ChromaDB (Vector Database)
-   LangChain Text Splitters

------------------------------------------------------------------------

## 🔬 Approach

### 1. Document Processing

-   Load document
-   Apply multiple chunking strategies:
    -   100 / 10
    -   300 / 50
    -   600 / 100
    -   1000 / 150

### 2. Embeddings

-   Generate embeddings using Titan model

### 3. Storage

-   Store embeddings in ChromaDB collections (one per strategy)

### 4. Retrieval

-   Query each collection
-   Retrieve Top-K chunks

### 5. Generation

-   Generate answers using LLM with retrieved context

------------------------------------------------------------------------

## 📊 Evaluation Framework

Used **LLM-as-a-Judge** approach with following metrics:

-   Correctness
-   Completeness
-   Relevance
-   Groundedness
-   Clarity

Each answer is scored from 1--5.

------------------------------------------------------------------------

## 📈 Sample Results

<img width="904" height="234" alt="image" src="https://github.com/user-attachments/assets/590d2f7a-7f18-472f-bd14-2bca6f194a5a" />

------------------------------------------------------------------------

## 💡 Key Insights

-   Very small chunks break context → poor answers
-   Very large chunks reduce retrieval precision
-   Mid-sized chunks with overlap perform best

------------------------------------------------------------------------

## 🧪 Learnings

-   RAG performance depends heavily on chunking
-   Evaluation is as important as implementation
-   LLM-as-judge is useful but requires careful parsing

------------------------------------------------------------------------

## 📌 Conclusion

This project demonstrates that **data processing choices (like
chunking)** significantly impact GenAI system performance, often more
than model choice.

------------------------------------------------------------------------

## 🔗 Author

Feroz
