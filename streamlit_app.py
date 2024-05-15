# import requests
# import streamlit as st

# # Function to interact with the chatbot API
# def query_api(query):
#     endpoint = 'http://localhost:5001/query'
#     data = {'query': query}
#     response = requests.post(endpoint, json=data)
#     if response.status_code == 200:
#         return response.json()['answer']
#     else:
#         return f"Error: {response.status_code}"

# # Streamlit UI
# st.title("Chatbot")

# query = st.text_input("Enter your query:")
# if st.button("Submit"):
#     if query:
#         answer = query_api(query)
#         st.text_area("Response:", value=answer, height=200)

import requests
import streamlit as st

# Function to interact with the chatbot API
def query_api(query):
    endpoint = 'http://localhost:5001/query'
    data = {'query': query}
    response = requests.post(endpoint, json=data)
    if response.status_code == 200:
        return response.json()['answer']
    else:
        return f"Error: {response.status_code}"

# Function to retrieve or create session state
def get_session_state():
    if 'session' not in st.session_state:
        st.session_state.session = {'chat_history': []}
    return st.session_state.session

# Streamlit UI
st.title("Chatbot")

# Retrieve or create session state
session_state = get_session_state()

query = st.text_input("Enter your query:")

if st.button("Submit"):
    if query:
        answer = query_api(query)
        session_state['chat_history'].append({'query': query, 'answer': answer})

for item in session_state['chat_history']:
    st.write(f"User: {item['query']}")
    st.write(f"Chatbot: {item['answer']}")
    st.write("----")
