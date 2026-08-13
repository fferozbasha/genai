import gradio as gr
import asyncio
import time
import json
from agents import Runner
import tutoragents
from utils.util_validate_answer import validate_answer
from schemas.feedback_request_schema import FeedbackRequestModel

# ---------- GLOBAL STATE ----------
state = {
    "intent": None,
    "current_q": 0,
    "total_q": 0,
    "results": [],
    "current_question": None,
    "start_time": None
}

# ---------- ASYNC HELPERS ----------
async def parse_intent(user_input):
    result = await Runner.run(tutoragents.user_intent_agent, user_input)
    return result.final_output

async def generate_valid_question(prompt):
    for _ in range(3):
        q_res = await Runner.run(tutoragents.question_generator_agent, prompt)
        question = q_res.final_output

        v_res = await Runner.run(
            tutoragents.question_validator_agent,
            question.model_dump_json()
        )

        if v_res.final_output.is_valid:
            return question

    raise ValueError("Failed to generate valid question")


async def get_feedback(question, is_correct, time_taken):
    feedback_request = FeedbackRequestModel(
        **question.model_dump(),
        time_taken=time_taken,
        is_correct=is_correct
    )

    res = await Runner.run(
        tutoragents.answer_feedback_agent,
        feedback_request.model_dump_json()
    )

    return res.final_output.feedback


async def get_final_feedback(results):
    res = await Runner.run(
        tutoragents.final_feedback_agent,
        json.dumps(results)
    )
    return res.final_output


# ---------- UI FUNCTIONS ----------

def start_session(user_input):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    intent = loop.run_until_complete(parse_intent(user_input))

    state["intent"] = intent
    state["current_q"] = 0
    state["total_q"] = intent.number_of_questions
    state["results"] = []

    return (
        f"Topic: {intent.topic} | Questions: {intent.number_of_questions}",
        *load_question()
    )


def load_question():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    intent = state["intent"]

    prompt = f"Generate a question on {intent.topic}"
    if hasattr(intent, "level") and intent.level:
        prompt += f" with {intent.level} difficulty"

    question = loop.run_until_complete(generate_valid_question(prompt))

    state["current_question"] = question
    state["start_time"] = time.time()

    return (
        question.question,
        gr.update(choices=question.choices, value=None),
        ""
    )


def submit_answer(user_answer):
    question = state["current_question"]
    time_taken = time.time() - state["start_time"]

    is_correct = validate_answer(user_answer, question.correct_answer)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    feedback = loop.run_until_complete(
        get_feedback(question, is_correct, time_taken)
    )

    record = {
        "question": question.question,
        "choices": question.choices,
        "correct_answer": question.correct_answer,
        "user_answer": user_answer,
        "is_correct": is_correct,
        "time_taken": time_taken,
        "feedback": feedback,
        "concept": question.concept
    }

    state["results"].append(record)
    state["current_q"] += 1

    if state["current_q"] < state["total_q"]:
        q_text, choices, _ = load_question()
        return q_text, choices, feedback

    else:
        # Final stage
        final_feedback = loop.run_until_complete(
            get_final_feedback(state["results"])
        )

        return "Completed!", gr.update(choices=[]), final_feedback


# ---------- UI LAYOUT ----------

with gr.Blocks() as app:

    gr.Markdown("## 🧠 Adaptive AI Maths Tutor")

    user_input = gr.Textbox(label="Enter your request")
    start_btn = gr.Button("Start")

    intent_display = gr.Textbox(label="Parsed Intent", interactive=False)

    question_text = gr.Textbox(label="Question", interactive=False)

    options = gr.Radio(choices=[], label="Select your answer")

    submit_btn = gr.Button("Submit Answer")

    feedback_box = gr.Textbox(label="Feedback", interactive=False)

    # ---------- EVENTS ----------

    start_btn.click(
        start_session,
        inputs=user_input,
        outputs=[intent_display, question_text, options, feedback_box]
    )

    submit_btn.click(
        submit_answer,
        inputs=options,
        outputs=[question_text, options, feedback_box]
    )

# ---------- RUN ----------
app.launch()