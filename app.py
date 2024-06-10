import streamlit as st
import requests
from transformers import pipeline

# Hugging Face model options
model_options = [
    "distilbert-base-uncased-distilled-squad",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "roberta-base-squad2",
    "deepset/roberta-base-squad2"
]

st.title("ReliefWeb Query App")

# Input for GLIDE number
glide_number = st.text_input("Enter GLIDE Number:")

disaster_info = None

if glide_number:
    # Query the ReliefWeb API to get disaster ID
    url = f"https://api.reliefweb.int/v1/disasters?appname=tom.jordan2@redcross.org&filter[field]=glide&filter[value]={glide_number}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['count'] > 0:
            disaster_id = data['data'][0]['id']
            # Query the ReliefWeb API to get disaster overview by ID
            overview_url = f"https://api.reliefweb.int/v1/disasters/{disaster_id}?appname=tom.jordan2@redcross.org"
            overview_response = requests.get(overview_url)
            if overview_response.status_code == 200:
                disaster_info = overview_response.json()['data'][0]['fields']
                st.write("### Disaster Information")
                st.json(disaster_info)
            else:
                st.write("Failed to fetch disaster overview from ReliefWeb API.")
        else:
            st.write("No data found for the provided GLIDE number.")
    else:
        st.write("Failed to fetch data from ReliefWeb API.")

# Text box for user questions
question = st.text_input("Ask a question based on the above information:")

# Dropdown for model selection
model_name = st.selectbox("Select a Hugging Face model for Q&A:", model_options)

if st.button("Get Answer"):
    if question and disaster_info:
        # Initialize the QA pipeline
        qa_pipeline = pipeline("question-answering", model=model_name)

        # Prepare context from disaster information
        context = "\n".join([f"{key}: {value}" for key, value in disaster_info.items()])

        # Get the answer
        result = qa_pipeline(question=question, context=context)
        st.write("### Answer")
        st.write(result['answer'])
    else:
        st.write("Please enter a question and ensure disaster information is available.")
