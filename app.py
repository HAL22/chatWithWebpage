import streamlit as st
from streamlit_chat import message
import chat

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

def generate_response(prompt,agent):
    st.session_state['messages'].append({"role": "user", "content": prompt})
    response = agent(prompt)['output']
    st.session_state['messages'].append({"role": "assistant", "content": response})
    return response

st.title("Chat placeholder")

# container for chat history
response_container = st.container()
# container for text box
container = st.container()
agent = chat.get_agent("https://en.wikipedia.org/wiki/Chelsea_F.C.")

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')     

    if submit_button and user_input:
        output = generate_response(user_input,agent)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))      
             
                    