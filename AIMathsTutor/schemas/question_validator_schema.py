from pydantic import BaseModel, Field
from typing import Optional

class MathsQuestionValidatorModel(BaseModel):
   
    is_valid: bool= Field(..., 
                         description="Flag to indicate if the question was valid or not based on the question and correct answer.")
    reason: Optional[str] = Field(None,
                                  description="Reason if the question is invalid")