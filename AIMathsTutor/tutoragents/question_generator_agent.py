from agents import Agent
from config import MODEL_NAME
from prompts.question_generator_prompt import QUESTION_GENERATOR_PROMPT
from schemas.question_generator_schema import MathsQuestionGeneratorModel

question_generator_agent = Agent(name="Question Generator",
                                 instructions=QUESTION_GENERATOR_PROMPT,
                                 model=MODEL_NAME,
                                 output_type=MathsQuestionGeneratorModel)