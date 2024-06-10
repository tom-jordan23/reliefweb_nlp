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

st.title("ReliefWeb NLP Query App")

# Input for GLIDE number
glide_number = st.text_input("Enter GLIDE Number:")

if glide_number:
    # Query the ReliefWeb API
    url = f"https://api.reliefweb.int/v1/disasters?appname=tom.jordan2@redcross.org&filter[field]=glide&filter[value]={glide_number}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['count'] > 0:
            disaster_info = data['data'][0]['fields']['profile']['overview']
            st.write("### Disaster Information")
            st.json(disaster_info)
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

        # Get the answer
        result = qa_pipeline(question=question, context=disaster_info)
        st.write("### Answer")
        st.write(result['answer'])
    else:
        st.write("Please enter a question and ensure disaster information is available.")

