# Adaptive AI Maths Tutor

## Overview

An AI-powered adaptive maths tutor built using OpenAI Agents SDK.\
This system generates questions, validates them, evaluates user answers,
and provides feedback.

------------------------------------------------------------------------

## Features

-   Multi-agent architecture (Intent, Generator, Validator, Feedback,
    Summary)
-   Structured outputs using Pydantic
-   Deterministic answer validation (no LLM hallucination)
-   Retry mechanism for invalid questions
-   Per-question feedback
-   Final performance summary
-   Gradio UI with MCQs

------------------------------------------------------------------------

## Tech Stack

-   Python
-   OpenAI Agents SDK
-   Pydantic
-   Gradio

------------------------------------------------------------------------

## Project Structure

    AIMathsTutor/
    │
    ├── tutoragents/
    ├── schemas/
    ├── prompts/
    ├── tools/
    ├── utils/
    ├── app.py
    ├── tutor.py
    ├── config.py
    └── requirements.txt

------------------------------------------------------------------------

## How to Run

### 1. Setup environment

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

### 2. Add API Key

Create `.env`:

    OPENAI_API_KEY=your_key_here
    MODEL_NAME=gpt-4.1-mini

### 3. Run app

    python app.py

------------------------------------------------------------------------

## Demo

(Add screenshot or video here)

------------------------------------------------------------------------


## Author

Feroz
