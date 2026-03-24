import boto3
import json
from botocore.config import Config

# Option to enable debug logging, if required. 
DEBUG = False

# Creating the bedrock client instance
bedrock = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",
    config=Config(read_timeout=3600),
)

# Empty messages list array. This will be used to store the conversation history
messages_list = []

# Just used to ensure setting some system prompts and additional info, only in the beginning of the chat
is_first_chat_message = True 

# loads the courts.json file
def load_courts():
    with open("data/courts.json") as f:
        return json.load(f)

# Infinite loop until the user exits the chat. 
while True:
    ai_response = ""
    user_input = input("You: ")

    # To rest the chat. Clears the conversation history and begins the chat from beginning
    if user_input == 'reset':
        messages_list = []
        is_first_chat_message = True
        print("Resetting the chat history ....")
        continue

    # Exits the chat completely
    if user_input == 'exit':
        is_first_chat_message = True
        print("Exiting the chat ....")
        break

    if is_first_chat_message:
        text  = "Please act as a badminton court booking agent"
        text += "\n. Understand the user's inputs can have any of the following intents. "
        text += "\n. Possible intents=filter_courts,get_cheapest_court,book_court"
        text +="\n . In each response from assistant, make sure to send the intent as a proper json."
        text += "\n. Provide the filters as an array."
        text += "\n. Remember that the user can try to filter based on any of the following fields only - name - ,location - string,sport - string , price_per_hour - float, available_slots - array of times (ex:5pm, 6am), rating - float, indoor - boolean, parking - boolean, rules - string"
        text += "\n. Do the filtering and other logics only based on the courts_list provided in the user input"
        text += "\n. At each assistants response, provide a field summary which should be a human readable text of whats been done."
        text += "\n. Return ONLY valid JSON. No extra text."
        text += "\n. Always return the fields intent, filters (json array), results (array), summary"

        courts = load_courts()
        # adding the fetched court details to the user prompt. 
        text += "\n. {\"courts_list\": " + str(courts) + " }" 

        text += user_input
        content = [{"text": text}]

        is_first_chat_message = False
    else:
        content = [{"text": {"User Query": user_input}}]

    messages_list.append({"role": "user", "content": content})

    # using the bedrock's converse_stream api to get the api respone in real-time like a stream
    response = bedrock.converse_stream(
        modelId="us.amazon.nova-2-lite-v1:0",
        messages = messages_list
    )

    if DEBUG:
        print("AI (Debug):", end=" ", flush=True)
        print("\n")

    # Stream response 
    for event in response["stream"]:
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]
            if "text" in delta:
                ai_response = ai_response + delta["text"]
                if DEBUG:
                    print(delta["text"], end="", flush=True)

    # The actual AI response has markdown. 
    # Clearing the markdown to make it proper json so that it can be parsed
    ai_response_clean = ai_response.strip()

    if ai_response_clean.startswith("```"):
        ai_response_clean = ai_response_clean.split("```")[1]  # remove ```
        if ai_response_clean.startswith("json"):
            ai_response_clean = ai_response_clean[4:]  # remove 'json'

    try:
        parsed_ai_response = json.loads(ai_response_clean)
    except:
        print("Failed to parse JSON. Raw response below:")
        print(ai_response)
        continue

    print(f"\n\nAI(Summary): {parsed_ai_response['summary']}")

    # Appending the conversation history
    messages_list.append({"role": "assistant", "content": [{"text": ai_response}]})

    print("\n") 
