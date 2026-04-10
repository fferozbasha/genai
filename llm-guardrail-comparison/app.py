import boto3
from openai import OpenAI
from botocore.config import Config

bedrock_modelId = 'us.amazon.nova-2-lite-v1:0'
bedrock_guardrail_identifier = 'y0iecl8f2h42'
debug = False # Set as True to print the LLM response as well

# OpenAI client instance
openAIClient = OpenAI()

# Creating the bedrock client instance
bedrock = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",
    config=Config(read_timeout=3600),
)

# Guardail config of the one created in AWS Bedrock
guardrailConfig={
        'guardrailIdentifier': bedrock_guardrail_identifier,
        'guardrailVersion': '1',
        'trace': 'enabled_full'
    }

# Invokes the Bedrock LLM Model. Includes the Guardrail config in the request
def get_llm_response_bedrock_with_guardrail(prompt=None):
    llm_response = bedrock.converse(
        modelId=bedrock_modelId,
        messages=[
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ],
        guardrailConfig=guardrailConfig
    )

    return llm_response

# Invokes teh Bedrock LLM Model without the Guardrail config. 
def get_llm_response_bedrock_without_guardrail(prompt=None):
    
    llm_response = bedrock.converse(
            modelId=bedrock_modelId,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ]
        )
    return llm_response

# Validates the OpenAI Moderation results to check for flagged categories
def validate_with_openai_moderation(content=None):

    flagged_categories = []
    results = []

    if not content:
        print("No content to validate for anything harmful or inappropriate")
        return
    
    # Invokes the OpenAI Moderation API
    moderation_results = openAIClient.moderations.create(
        model="omni-moderation-latest",
        input=content
    )

    # Transforms the model results which is in Object type to a dict version. 
    moderation_results_json = moderation_results.model_dump()["results"][0]

    # If the flagged attribute is true, it means one or more categories has 
    # high confidence score of being part of the content. 
    if moderation_results_json.get("flagged", None) == True:
        categories = moderation_results_json.get("categories", None)

        # Retrieves the list of flagged categories. 
        for category, value in categories.items():
            if value == True:
                flagged_categories.append(category)

        # For the flagged categories, gets the confidence score. 
        category_scores = moderation_results_json.get("category_scores", None)
        for category, score in category_scores.items():
            if category in flagged_categories:
                results.append(f"{category.upper()} : {round(score * 100, 1)}% confidence")

        return results
    else:
        return []


def extract_guardrail_summary(llm_response, assessmentLevel='inputAssessment'):

    """
    Used to extract the assessment details from the Guardrail trace. 
    Can be used to extract the details either for input or output assessment
    Checks for topicPolicy, contentPolicy and sensitiveInformationPolicy details. 
    Wherever a policy is detected, corresponding policy details are appended to 
    the final result array. 
    """

    trace = llm_response.get('trace', {})

    if assessmentLevel not in ['inputAssessment', 'outputAssessment']:
        print(f"Guardrail assessment can be done either for Input or for Output only")

    result = []
    blocked_level = None

    guardrail = trace.get("guardrail", {})
    assessments = guardrail.get(assessmentLevel, {}).get(bedrock_guardrail_identifier, {})

    if assessmentLevel == 'inputAssessment':
        blocked_level = 'User Prompt'
    else:
        blocked_level = 'LLM Response'

    for policyName , policies in assessments.items():

        if policyName == 'topicPolicy':
            topics = policies.get('topics', {})
            for topic in topics:
                if topic.get('detected') == True:
                    result.append(f"{blocked_level} blocked by Policy: Topic, Reason: {topic.get('name')}")

        if policyName == 'contentPolicy':
            filters = policies.get('filters', {})
            for filter in filters:
                if filter.get('detected') == True:
                    result.append(f"{blocked_level} blocked by Policy: Content, Reason: {filter.get('type')}")
                    
        if policyName == 'sensitiveInformationPolicy':
            piiEntities = policies.get('piiEntities')
            for piiEntity in piiEntities:
                if piiEntity.get('detected') == True:
                    result.append(f"{blocked_level} blocked by Policy: PII Information, Reason: {piiEntity.get('type')}")

    return result

def extract_bedrock_llm_response_output(rawLLMResponse):
    return rawLLMResponse["output"]["message"]["content"][0]["text"]

# Gets the User input promp
user_input = input("You: ")

print("\n")
print("Starting with LLM Invocations")
print("=============================")
print("\n")

print("Invoking Bedrock LLM without Guardrail")
bedrock_llm_response_without_guardrail = get_llm_response_bedrock_without_guardrail(user_input)
print(f"Response Time      = {bedrock_llm_response_without_guardrail["metrics"]["latencyMs"]} ms")
print(f"Total Token count  = {bedrock_llm_response_without_guardrail["usage"]["totalTokens"]}")
if debug:
    print(extract_bedrock_llm_response_output(bedrock_llm_response_without_guardrail))


print("----------------------------------------------------------------------------------")
print("Invoking Bedrock LLM with Guardrail")
bedrock_llm_response_with_guardrail = get_llm_response_bedrock_with_guardrail(user_input)

print(f"Response Time      = {bedrock_llm_response_with_guardrail["metrics"]["latencyMs"]} ms")
print(f"Total Token count  = {bedrock_llm_response_with_guardrail["usage"]["totalTokens"]}")

if debug:
    print(extract_bedrock_llm_response_output(bedrock_llm_response_with_guardrail))

input_guardrail_assessment = extract_guardrail_summary(
    bedrock_llm_response_with_guardrail, 
    assessmentLevel='inputAssessment')

output_guardrail_assessment = extract_guardrail_summary(
    bedrock_llm_response_with_guardrail, 
    assessmentLevel='outputAssessment')

stop_reason = bedrock_llm_response_with_guardrail['stopReason']

if stop_reason == 'guardrail_intervened':
    print(f"LLM Response : {stop_reason}")
    if input_guardrail_assessment:
        print("\n".join(input_guardrail_assessment))
    if output_guardrail_assessment:
        print("\n".join(output_guardrail_assessment))

print("----------------------------------------------------------------------------------")

print("Validating User Prompt with OpenAI Moderation")
user_input_moderation_results = validate_with_openai_moderation(user_input)
if user_input_moderation_results:
    print(f"User Input has blocked content as per Moderation API")
    print("\n".join(user_input_moderation_results))
else:
    print("Re-using the LLM response received earlier from Bedrock to do output moderation validation")
    llm_response_moderation_results = validate_with_openai_moderation(extract_bedrock_llm_response_output(bedrock_llm_response_without_guardrail))
    if llm_response_moderation_results:
        print(f"LLM Response has blocked content as per Moderation API")
        print("\n".join(llm_response_moderation_results))
    else:
        print("LLM response found harmless as per OpenAI moderation API")