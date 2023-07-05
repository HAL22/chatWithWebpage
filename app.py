import streamlit as st
from streamlit_chat import message
from chat import *

# Page title
st.set_page_config(page_title='🦜🔗 Chat With Web Page')
st.title('🦜🔗 Chat with any webpage')

get_agent_container = st.container()
input_container = st.container()
#colored_header(label='', description='', color_name='blue-30')
response_container = st.container()

# Text input
def get_url_user():
    txt_input = st.text_area('Enter url', '', height=80)
    return txt_input
with get_agent_container:
    agent = get_agent(get_url_user())

def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text
## Applying the user input box
with input_container:
    user_input = get_text()    

def generate_response(prompt):
    reponse = agent(prompt)['output']

## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if user_input:
        response = generate_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)
        
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))