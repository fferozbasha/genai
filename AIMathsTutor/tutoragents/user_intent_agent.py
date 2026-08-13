from agents import Agent
from config import MODEL_NAME
from prompts.intent_response_prompt import INTENT_RESPONSE_PROMPT
from schemas.intent_response_schema import UserIntentModel

user_intent_agent = Agent(name="Extract User Intent",
                                 instructions=INTENT_RESPONSE_PROMPT,
                                 model=MODEL_NAME,
                                 output_type=UserIntentModel)