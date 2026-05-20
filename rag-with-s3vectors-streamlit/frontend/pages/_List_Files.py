import streamlit as st
import boto3
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

st.title("List Source Files")

dynamodb = boto3.resource('dynamodb')
audit_table = dynamodb.Table('s3-vector-chunks-entry')

def get_all_items_dynamodb():
    all_items = []
    last_returned_key = None

    while True:
        # If last_evaluated_key is present, continue from where we left off
        if last_returned_key:
            response = audit_table.scan(ExclusiveStartKey=last_returned_key)
        else:
            response = audit_table.scan()

        all_items.extend(response.get('Items', []))
        
        # Check if there are more items to retrieve
        last_returned_key = response.get('LastEvaluatedKey')
        if not last_returned_key:
            print("returning all items")
            return all_items

table_rows = []
items = get_all_items_dynamodb()

total_pii_count = 0

for item in items:

    pii_summary = " | ".join([
        f"{pii_entity[:3]}:{int(count)}"
        for pii_entity, count in item["pii_summary"].items()
    ])

    total_pii_count += sum([
        int(count)
        for count in item["pii_summary"].values()
    ])

    local_time = (
        datetime.fromisoformat(item["timestamp"])
        .replace(tzinfo=ZoneInfo("UTC"))
        .astimezone(ZoneInfo("Australia/Melbourne"))
        .strftime("%d-%b-%Y %I:%M %p")
    )

    table_rows.append({
        "📄 File Name": item["s3_object_name"],
        "🕒 Timestamp": local_time,
        "🧠 Vectors": int(
            item["total_vectors_inserted"]
        ),
        "🔒 PII Summary": pii_summary
    })

df = pd.DataFrame(table_rows)

df = df.sort_values(
    by="🕒 Timestamp",
    ascending=False
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "📄 Total Files",
    len(df)
)

col2.metric(
    "🧠 Total Vectors",
    int(df["🧠 Vectors"].sum())
)

col3.metric(
    "🔒 Total PII",
    total_pii_count
)

st.divider()

st.data_editor(
    df,
    use_container_width=True,
    hide_index=True,
    height=250,
    disabled=True
)

# for item in items:

#     pii_summary = "\n".join([
#         f"{pii_entity}: {int(count)}"
#         for pii_entity, count in item["pii_summary"].items()
#     ])

#     local_time = (
#         datetime.fromisoformat(item["timestamp"])
#         .replace(tzinfo=ZoneInfo("UTC"))
#         .astimezone(ZoneInfo("Australia/Melbourne"))
#         .strftime("%d-%b-%Y %I:%M %p")
#     )

#     table_rows.append({
#     "File Name": item["s3_object_name"],
#     "Timestamp": local_time,
#     "Vectors Inserted": int(
#         item["total_vectors_inserted"]
#     ),
#     "PII Summary": pii_summary
# })

    
# df = pd.DataFrame(table_rows)

# df = df.sort_values(
#     by="Timestamp",
#     ascending=False
# )

# st.data_editor(
#     df,
#     use_container_width=True,
#     hide_index=True,
#     height=400,
#     disabled=True
# )