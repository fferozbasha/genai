from agents import Agent
from config import MODEL_NAME
from prompts.feedback_response_prompt import ANSWER_FEEDBACK_PROMPT
from schemas.feedback_response_schema import MathsAnswerFeedbackModel

answer_feedback_agent = Agent(name="Answer Feedback",
                                 instructions=ANSWER_FEEDBACK_PROMPT,
                                 model=MODEL_NAME,
                                 output_type=MathsAnswerFeedbackModel)