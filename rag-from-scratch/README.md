# RAG From Scratch using AWS Bedrock (Titan Embeddings + Nova)

## 🚀 Overview
This project demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline built completely from scratch using Python — without using frameworks like LangChain.

It showcases how to:
- Generate embeddings using AWS Bedrock (Titan Embeddings)
- Perform similarity search using cosine similarity
- Retrieve top relevant documents
- Augment prompt with context
- Generate responses using a Foundation Model (Nova)

---

## 🧠 Architecture

1. Input documents (in-memory list)
2. Generate embeddings for each document
3. Store embeddings alongside text
4. Accept user query
5. Convert query into embedding
6. Compute cosine similarity
7. Filter + rank documents
8. Select top-k relevant documents
9. Augment prompt with retrieved context
10. Send to LLM → get response

---

## 🛠️ Tech Stack

- Python
- AWS Bedrock
  - Titan Embeddings (amazon.titan-embed-text-v2)
  - Nova Model (us.amazon.nova-2-lite-v1)
- NumPy (for cosine similarity)

---

## 📦 Installation

```bash
pip install boto3 numpy
```

---

## 🔧 Configuration

Make sure AWS credentials are configured:

```bash
aws configure
```

Ensure access to:
- Bedrock Runtime
- Titan Embedding model
- Nova model

---

## ▶️ How to Run

```bash
python app.py
```

Then enter your query when prompted.

---

## 📌 Example Flow

<img width="1488" height="366" alt="image" src="https://github.com/user-attachments/assets/37c969c2-3258-468f-bdad-aa9c2f2b8eab" />

---

## 💡 Key Learnings

- RAG is fundamentally:
  - Embeddings + Similarity Search + Context Injection
- Vector databases are optional for small datasets
- Understanding the basics helps avoid over-reliance on frameworks

---

## ⚠️ Notes

- This is a **learning/demo project**, not production-ready
- Embeddings are generated per request (no caching yet)
- No vector DB used (in-memory storage)

---

## 🔥 Why This Project

Most RAG tutorials rely on frameworks.

This project helps you:
- Understand **what actually happens under the hood**
- Build confidence in designing your own AI systems

---

## 👨‍💻 Author

Built as part of a journey to deeply understand **Generative AI and RAG systems**.

