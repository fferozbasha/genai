from pydantic import BaseModel, Field, model_validator

class MathsQuestionGeneratorModel(BaseModel):
   
    question: str= Field(..., 
                         description="Properly worded maths question")
    
    correct_answer: str = Field(..., 
                                description="Correct answer that MUST match exactly one of the choices")
    
    choices: list[str] = Field(..., 
                               min_length=4,
                               max_length=4,
                               description="Exactly 4 answer choices including the correct answer")
    
    concept: str = Field(..., 
                         description="One or two word concept tested (e.g., Fractions, Area)")

    @model_validator(mode="after")
    def check_answer_in_choices(self):
        if self.correct_answer not in self.choices:
            raise ValueError("correct_answer must be in choices")
        return self