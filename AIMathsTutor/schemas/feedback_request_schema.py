from schemas.question_generator_schema import MathsQuestionGeneratorModel
from pydantic import Field

class FeedbackRequestModel(MathsQuestionGeneratorModel):
    time_taken: float = Field(description="Time taken for the user to answer the question")
    is_correct: bool = Field(description="Flag to indicate if the users answer was correct or incorrect.")