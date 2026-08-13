
import time
import json
import uuid
import asyncio
from agents import Runner, trace

from utils.util_validate_answer import validate_answer
from schemas.feedback_request_schema import FeedbackRequestModel
import tutoragents

#base method
async def runagent(user_input:str):

    session_id = str(uuid.uuid4())
    with trace(session_id):
        user_intent = await Runner.run(
            tutoragents.user_intent_agent,
            user_input
        )

        user_intent_response = user_intent.final_output

        print(f"Topic for practice: {user_intent_response.topic}")
        print(f"Number of questions to practice: {user_intent_response.number_of_questions}")

        question_generator_prompt = f"Generate a question on topic {user_intent_response.topic}"

        if user_intent_response.level:
            print(f"Difficult level of questions: {user_intent_response.level}")
            question_generator_prompt+= f" with difficulty level of {user_intent_response.level}"


        print("----------------------------------------------------------------")

        results = []

        for i in range(user_intent_response.number_of_questions):

            is_generated_question_valid = False
            count_question_generator = 0
            question_generated = None

            while(not is_generated_question_valid):

                if(count_question_generator == 2):
                    raise ValueError("Too many invalid questions")

                question_generator_result = await Runner.run(
                    tutoragents.question_generator_agent,
                    question_generator_prompt
                )

                question_generated = question_generator_result.final_output

                #Validate the question generated
                question_validator_result = await Runner.run(
                    tutoragents.question_validator_agent,
                    question_generated.model_dump_json()
                )

                is_generated_question_valid = question_validator_result.final_output.is_valid

                if not is_generated_question_valid:
                    print(f"Question: {question_generated} is invalid because {question_validator_result.final_output.reason}")

                count_question_generator += 1

            start_time = time.time()
            print("----------------------------------------------------------------")
            print(f"Question {i}: {question_generated}")
            user_answer = input("Please enter your answer:\n")
            time_taken = time.time() - start_time
            if user_answer:
                is_correct = validate_answer(user_answer, question_generated.correct_answer)

                feedback_request = FeedbackRequestModel(
                    **question_generated.model_dump(),  
                    time_taken=time_taken,
                    is_correct=is_correct
                )

                feedback = await Runner.run(
                    tutoragents.answer_feedback_agent,
                    f"Generate feedback for:\n{feedback_request.model_dump_json()}"
                )

                record = {
                    "question": question_generated.question,
                    "choices": question_generated.choices,
                    "correct_answer": question_generated.correct_answer,
                    "user_answer": user_answer,
                    "is_correct": is_correct,
                    "time_taken": time_taken,
                    "feedback": feedback.final_output.feedback,
                    "concept": question_generated.concept
                }

                results.append(record)


        final_feedback = await Runner.run(
            tutoragents.final_feedback_agent,
            f"Analyze the following results:\n{json.dumps(results)}"
        )

        print("Final Feedback")
        print(final_feedback.final_output)
        
#user_input = input()
user_input = "Generate 3 maths question for Grade 4 level student with ICAS level difficulty"
asyncio.run(runagent(user_input))
