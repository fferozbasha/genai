import streamlit as st
import boto3
import time
import pandas as pd

st.title("Upload Document")

S3_BUCKET_NAME = "feroz-s3-rag-source-documents"
STEP_FUNC_ARN = "arn:aws:states:ap-southeast-2:414079114528:stateMachine:rag-s3-state-machine-pipeline"

STEP_FUNC_TASK_NAME_DETECT_PII_ENTITIES = "Detect PII Entities and Redact the docuemnt"
STEP_FUNC_TASK_NAME_INSERT_S3_VECTORS   = "Insert in to S3 Vectors"

s3 = boto3.client("s3")
dynamodb = boto3.resource('dynamodb')
stepfunctions = boto3.client("stepfunctions")
audit_table = dynamodb.Table('s3-vector-chunks-entry')
step_funtion_running_execution = None

# File uploader to allow the user to upload the file.
uploaded_file = st.file_uploader(
    "Upload your fraud document",
    type=["txt"]
)

# If user has uploaded a file successfully
if uploaded_file:
    try:
        # Start uploading the file to the S3 Bucket.
        s3_upload_status = s3.upload_fileobj(
        uploaded_file,
        S3_BUCKET_NAME,
        uploaded_file.name
        )

        st_status_file_uploaded = st.success(f"File {uploaded_file.name} uploaded successfully to S3")

    except Exception as ex:
        st_status_file_uploaded = st.error(F"File upload to S3 failed: {str(ex)}")
        st.stop()

    # Checking in loop to up to 3 times, while waiting for 3 seconds each time to 
    # check if the event bridge has triggered the Step function to upload the 
    # document to S3 Vector database. 
    with st.spinner("Waiting for ingestion pipeline to start..."):
        for _ in range(3):
            step_functions_response = stepfunctions.list_executions(
                stateMachineArn=STEP_FUNC_ARN,
                statusFilter="RUNNING",
                maxResults=1
            )

            step_function_executions = step_functions_response["executions"]

            # If able to find any recent running execution, then can assume that the
            # process has started successfully. 
            if step_function_executions:
                st.success("RAG Pipeline to update the S3 Vector database has started")
                step_funtion_running_execution = step_function_executions[0]
                break

            time.sleep(3)

    if not step_funtion_running_execution:
        st.error(f"Unable to find any Step function execution running yet.")
        st.stop()

    step_funtion_running_execution_arn = step_funtion_running_execution['executionArn']

    # Setting initial status of the Step Function execution ARN to 'Running'
    step_funtion_running_execution_status = "RUNNING"

    with st.spinner("RAG Pipeline to update the S3 Vector database is running..."):

        while step_funtion_running_execution_status == "RUNNING":

            response = stepfunctions.describe_execution(
                executionArn=step_funtion_running_execution_arn
            )

            step_funtion_running_execution_status = response["status"]

            time.sleep(3)

    if step_funtion_running_execution_status == "SUCCEEDED":
        st.success("Document Ready to Query")
        audit_table_results = audit_table.get_item(Key={'s3_object_name': uploaded_file.name})
        item = audit_table_results["Item"]

        st.write("## Upload Summary")
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Vectors Inserted",
            item["total_vectors_inserted"]
        )

        col2.metric(
            "Replaced Vectors ?",
            str(item["is_existing_vectors_deleted"])
        )

        col3.metric(
            "Total PII Found",
            sum(item["pii_summary"].values())
        )

        st.write("### PII Summary")
        for pii_type, count in item["pii_summary"].items():
            st.write(f"- {pii_type}: {int(count)}")

    else:
        st.error(f"Pipeline ended with status: {step_funtion_running_execution_status}")
