# 🤖 Adaptive AI Maths Tutor

## 📌 Overview

An AI-powered adaptive maths tutor built using **OpenAI Agents SDK**.

This system simulates a real tutor by: - Generating questions -
Validating quality - Evaluating answers - Providing feedback -
Summarising performance

------------------------------------------------------------------------

## 🔄 How It Works (Flow)

    User Input
       ↓
    Intent Agent
       ↓
    Question Generator Agent
       ↓
    Validator Agent
       ↓
    User Answer
       ↓
    Answer Validation (Function)
       ↓
    Feedback Agent
       ↓
    Session Tracking
       ↓
    Summary Agent

------------------------------------------------------------------------

## ⚙️ Step-by-Step Flow

### 1. User Input

User provides: - Topic - Grade level - Exam type (ICAS / NAPLAN) -
Number of questions

------------------------------------------------------------------------

### 2. Intent Agent

Parses user input into structured format: - Topic - Difficulty - Number
of questions

------------------------------------------------------------------------

### 3. Question Generator Agent

-   Generates **one MCQ question at a time**
-   Uses **Pydantic structured output**
-   Includes:
    -   Question
    -   4 choices
    -   Correct answer
    -   Concept

------------------------------------------------------------------------

### 4. Validator Agent

-   Ensures question quality
-   Checks:
    -   Logical correctness
    -   Answer exists in choices
-   Retries if invalid (max 3 times)

------------------------------------------------------------------------

### 5. User Interaction

-   User answers MCQ
-   Time taken is captured

------------------------------------------------------------------------

### 6. Answer Validation (Deterministic)

-   No LLM used
-   Simple function compares answers
-   Ensures **100% correctness validation**

------------------------------------------------------------------------

### 7. Feedback Agent

Provides: - Explanation - Mistake insights - Reinforcement

------------------------------------------------------------------------

### 8. Session Tracking

Stores: - Question - User answer - Correct answer - Feedback - Time
taken - Concept

------------------------------------------------------------------------

### 9. Summary Agent

Generates: - Strengths - Weak areas - Behaviour insights

------------------------------------------------------------------------

## 🧠 Key Learnings

-   Structured outputs using **Pydantic**
-   Multi-agent orchestration
-   LLM validation patterns
-   Reliable AI system design
-   Tracing and observability

------------------------------------------------------------------------

## 🛠 Tech Stack

-   Python
-   OpenAI Agents SDK
-   Pydantic
-   Gradio

------------------------------------------------------------------------

## 📁 Project Structure

    AIMathsTutor/
    ├── tutoragents/
    ├── schemas/
    ├── prompts/
    ├── utils/
    ├── app.py
    ├── tutor.py
    ├── config.py
    └── requirements.txt

------------------------------------------------------------------------

## 🚀 How to Run

### 1. Setup

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

### 2. Configure

Create `.env` file:

    OPENAI_API_KEY=your_key
    MODEL_NAME=gpt-4.1-mini

### 3. Run

    python app.py

------------------------------------------------------------------------

## Demo

<img width="1103" height="656" alt="image" src="https://github.com/user-attachments/assets/97b00dba-8970-4b4c-b104-0963156a846d" />

