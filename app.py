import streamlit as st
import chat

def show_messages(text):
    messages_str = [
        f"{_['role']}: {_['content']}" for _ in st.session_state["messages"][1:]
    ]
    text.text_area("Messages", value=str("\n".join(messages_str)), height=400)


st.header("STREAMLIT GPT-3 CHATBOT")

web_url = st.text_input("Prompt", value="Enter the url here...")

agent_created = False

if st.button("Generate agent"):
    if web_url != "":
        with st.spinner("Generating response..."):
            agent = chat.get_agent(web_url)
            agent_created = True


prompt = st.text_input("Prompt", value="Enter your message here...")

if st.button("Send") and agent_created:
    with st.spinner("Generating response..."):
        output = agent(prompt)['output']
        show_messages(output)
