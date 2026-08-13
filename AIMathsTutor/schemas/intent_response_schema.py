from pydantic import BaseModel, Field
from typing import Optional

class UserIntentModel(BaseModel):
    topic: str = Field(...,
                       description="Math topic for which the user wants the question to be generated")
    number_of_questions: int = Field(...,
                                     ge=1,
                                     le=10,
                                     description="Total number of questions does the want to get generated")
    level:Optional[str] = Field(None, 
                                description="Difficult level of the questions to be generated")
    
    