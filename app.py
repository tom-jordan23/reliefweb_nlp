import streamlit as st
import requests
from transformers import pipeline
import openai
import os

# Hugging Face model options
model_options = {
    "distilbert-base-uncased-distilled-squad": "distilbert-base-uncased-distilled-squad",
    "bert-large-uncased-whole-word-masking-finetuned-squad": "bert-large-uncased-whole-word-masking-finetuned-squad",
    "deepset/roberta-base-squad2": "deepset/roberta-base-squad2",
    "nlp-thedeep/humbert": "nlp-thedeep/humbert",
    "chatgpt-rag": "ChatGPT with RAG"
}

st.title("ReliefWeb Query App")

# Add an overview and instructions
st.markdown("""
### Overview
This app allows you to query the ReliefWeb API using a GLIDE number to retrieve disaster information. You can then ask questions based on this information using various Hugging Face machine learning models.

### Instructions
1. Enter a GLIDE number in the input field (e.g. FL-2024-000036-BRA or TC-2024-000083-BGD)
2. The app will fetch and display the disaster overview associated with the GLIDE number.
3. Enter a question related to the displayed information.
4. Select a machine learning model from the dropdown menu. Note that the first time using a model will take a bit.
5. Click the 'Get Answer' button to receive an answer based on the selected model and the retrieved disaster information.
""")

# load my openai api key from a file
openai_key_path = '/home/tjordan/code/secrets/openai'
with open(openai_key_path, 'r') as file:
    # Read the contents of the file into a variable
    OPENAI_API_KEY = file.read()


# Function to load models and cache them
@st.cache_resource
def load_model(model_name):
    return pipeline("question-answering", model=model_name)

# Pre-load models
models = {name: load_model(name) for name in model_options.values() if name != "ChatGPT with RAG"}

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
                disaster_info = overview_response.json()
                overview = disaster_info['data'][0]['fields']['profile']['overview']
                st.write("### Disaster Overview")
                st.write(overview)
            else:
                st.write("Failed to fetch disaster overview from ReliefWeb API.")
        else:
            st.write("No data found for the provided GLIDE number.")
    else:
        st.write("Failed to fetch data from ReliefWeb API.")

# Text box for user questions
question = st.text_input("Ask a question based on the above information:")

# Dropdown for model selectionopena
model_name = st.selectbox("Select a model for Q&A:",list(model_options.keys()))

if st.button("Get Answer"):
    if question and disaster_info:
        if model_name == "chatgpt-rag":
            # Use ChatGPT with RAG approach
            with st.spinner('Loading the model...'):
                combined_input = f"Context: {overview}\n\nQuestion: {question}"
                
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                            {"role": "system", "content": "You are a helpful assistant."},  # System message to set the assistant's behavior
                            {"role": "user", "content": combined_input}],  # User message with the query and context
                    max_tokens=150,
                    api_key = OPENAI_API_KEY
                )
                answer = response.choices[0]['message']['content']
            
            st.write("### Answer")
            st.write(answer)
        else:
            with st.spinner("Running inference..."):
                # Initialize the QA pipeline
                qa_pipeline = pipeline("question-answering", model=model_name)

                # Get the answer
                result = qa_pipeline(question=question, context=overview)
                st.write("### Answer")
                st.write(result['answer'])
    else:
        st.write("Please enter a question and ensure disaster information is available.")
