from agents import Agent
from config import MODEL_NAME
from prompts.final_feedback_prompt import FINAL_FEEDBACK_PROMPT

final_feedback_agent = Agent(name="Final Feedback",
                                 instructions=FINAL_FEEDBACK_PROMPT,
                                 model=MODEL_NAME)