import streamlit as st
from streamlit_chat import message
import chat

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

def generate_response():
    prompt = st.session_state.user_input
    agent = chat.get_agent("https://en.wikipedia.org/wiki/Chelsea_F.C.")
    st.session_state['messages'].append({"role": "user", "content": prompt})
    response = agent(prompt)['output']
    st.session_state['messages'].append({"role": "assistant", "content": response})

st.title("Chat placeholder")

chat_placeholder = st.empty()

if st.session_state['generated']:
    with chat_placeholder:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))

with st.container():
    st.text_input("User Input:", on_change=generate_response, key="user_input")            