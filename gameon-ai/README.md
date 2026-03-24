# 🏸 GameOn AI Copilot (AWS Bedrock)

A **context-aware conversational assistant** built using AWS Bedrock to
explore **multi-turn interactions, streaming responses, and structured
LLM outputs**.

This project focuses on understanding how to **orchestrate LLMs in real
applications**, rather than just building a basic chatbot.

------------------------------------------------------------------------

## 🚀 What this project demonstrates

-   💬 Multi-turn conversational AI using `converse_stream`
-   ⚡ Real-time streaming responses (token-by-token output)
-   🧠 Conversation memory using structured message history
-   🧾 Structured JSON outputs (intent, filters, results, summary)
-   🎯 LLM-driven intent understanding (no hardcoded rules)
-   📊 Domain-aware responses using custom dataset (`courts.json`)

------------------------------------------------------------------------

## 🧠 Problem it solves

Instead of simple Q&A, the assistant can:

Find courts under \$20\
Which one is cheapest?\
Book it

It maintains context across turns and responds intelligently based on
prior conversation.

------------------------------------------------------------------------

## 🏗️ How it works

User Input\
↓\
Conversation History (messages list)\
↓\
AWS Bedrock (Nova model via converse_stream)\
↓\
Structured JSON Response (intent + filters + results + summary)\
↓\
Application parses response → displays summary

------------------------------------------------------------------------

## 📦 Tech Stack

-   Python 3.12\
-   AWS Bedrock
    -   `converse_stream` API\
    -   Amazon Nova model\
-   boto3

------------------------------------------------------------------------

## 📁 Project Structure

gameon-ai/\
│\
├── app.py\
├── data/\
│ └── courts.json\
├── .venv/\
└── README.md

------------------------------------------------------------------------

## ⚙️ Setup

### 1. Clone repository

git clone `<your-repo-url>`{=html}\
cd gameon-ai

------------------------------------------------------------------------

### 2. Create virtual environment

python3.12 -m venv .venv\
source .venv/bin/activate

------------------------------------------------------------------------

### 3. Install dependencies

pip install boto3 awscli

------------------------------------------------------------------------

### 4. Configure AWS

aws configure

Ensure:\
- Region: `us-east-1`\
- Bedrock model access enabled

------------------------------------------------------------------------

## ▶️ Run the app

python app.py

------------------------------------------------------------------------

## 🧪 Example interaction

You: cheapest court

AI (Summary): I found the cheapest available court:\
- Melbourne Sports Centre (\$18)

You: book it

AI (Summary): Successfully booked Melbourne Sports Centre at 7pm 🎉

------------------------------------------------------------------------

## 🧠 Key learnings

-   LLMs can be guided to produce **structured outputs (JSON)**
-   Multi-turn context enables **more natural interactions**
-   Streaming APIs improve **user experience significantly**
-   Prompt design is critical for **controlling model behavior**
-   LLM-only logic works for prototypes but needs **hybrid approaches
    for production**

------------------------------------------------------------------------

## ⚠️ Limitations

-   Relies fully on LLM for decision-making (non-deterministic)
-   JSON responses may occasionally require cleaning/parsing
-   Full dataset is injected in prompt (not scalable)
-   No backend validation or persistence

------------------------------------------------------------------------

## 🚀 Future improvements

-   Add Python-based filtering (hybrid LLM + deterministic logic)
-   Introduce tool/function calling
-   Implement RAG with embeddings
-   Add UI (Streamlit / web app)
-   Improve robustness and error handling

------------------------------------------------------------------------

## 👤 About this project

This project was built as part of learning AWS Bedrock and exploring how
to move from:

simple prompt-response → structured, context-aware AI applications

------------------------------------------------------------------------
