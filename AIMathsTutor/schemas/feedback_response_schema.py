from pydantic import BaseModel, Field

class MathsAnswerFeedbackModel(BaseModel):
   
    feedback: str= Field(..., 
                         max_length=500,
                         description="Feedback for the user based on the question, correct answer, users answer and the concept.")

    