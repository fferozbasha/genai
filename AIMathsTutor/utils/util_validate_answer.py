
def validate_answer(user_answer:str, correct_answer:str) -> bool:
    """
    Use this tool to validae the Users answer against the Correct answer.
    """
    if not user_answer or not correct_answer:
        raise("User answer and the correct answer to validate are mandatory")
        return False

    if(user_answer == correct_answer):
        #print("User answer matches the correct answer")
        return True
    else:
        #print("User answer does not match the correct answer")
        return False