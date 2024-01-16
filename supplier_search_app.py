import streamlit as st
import pandas as pd
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

def upload_csv():
    """
    Upload a CSV file and return its data.
    """
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if not uploaded:
        return None
    for fn in uploaded:
        return pd.read_csv(uploaded)

def create_prompt_from_csv(csv_data):
    """
    Write a prompt based on the CSV data.
    """
    prompt = "Based on the following data:\n\n"
    prompt += csv_data.to_string(index=False)
    prompt += "\n\n Suggest me the list of suppliers in India location providing 200-300 cotton gsm with a payment period of 90 days"
    return prompt

def run_clarifai_inference(prompt):
    """
    Run Clarifai inference using the given prompt.
    """
    PAT = '54a0ed80ce094bfca2cd6d53a51911c0'
    USER_ID = 'meta'
    APP_ID = 'Llama-2'
    MODEL_ID = 'llama2-70b-chat'
    MODEL_VERSION_ID = 'acba9c1995f8462390d7cb77d482810b'

    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)
    metadata = (('authorization', 'Key ' + PAT),)
    userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)

    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=userDataObject,
            model_id=MODEL_ID,
            version_id=MODEL_VERSION_ID,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        text=resources_pb2.Text(
                            raw=prompt
                        )
                    )
                )
            ]
        ),
        metadata=metadata
    )

    if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
        raise Exception(f"Post model outputs failed, status: {post_model_outputs_response.status.description}")

    output = post_model_outputs_response.outputs[0]
    return output.data.text.raw

def main():
    st.title("Clarifai Streamlit App")

    csv_data = upload_csv()

    if csv_data is not None:
        st.dataframe(csv_data)

        prompt = create_prompt_from_csv(csv_data)
        user_input = st.text_area("Generated Prompt", prompt)

        if st.button("Run Clarifai Inference"):
            output = run_clarifai_inference(user_input)
            st.text("Clarifai Inference Output:")
            st.write(output)

if __name__ == "__main__":
    main()
