from agents import Agent
from config import MODEL_NAME
from prompts.question_validator_prompt import QUESTION_VALIDATOR_PROMPT
from schemas.question_validator_schema import MathsQuestionValidatorModel

question_validator_agent = Agent(name="Question Validator",
                                 instructions=QUESTION_VALIDATOR_PROMPT,
                                 model=MODEL_NAME,
                                 output_type=MathsQuestionValidatorModel)