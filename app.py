import streamlit as st
import requests
import tiktoken
import pinecone
import os
from langchain import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from bs4 import BeautifulSoup

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

pinecone.init(
    api_key=st.secrets['PINECONE_API_KEY'],
    environment=st.secrets['PINECONE_ENV']
)

def html_to_text(url):
    r = requests.get(url)
    return r.text


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding('cl100k_base')
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

def clean_text(text):
    return BeautifulSoup(text, "lxml").text

def split_text(text):

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,  # number of tokens overlap between chunks
    length_function=tiktoken_len,
    separators=['\n\n', '\n', ' ', '']
    )

    return text_splitter.split_text(text)

def create_document_from_webpage(url):

    text = clean_text(html_to_text(url))

    texts = split_text(text)

    docs = [Document(page_content=t) for t in texts]

    return docs

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def creat_embeddings(url):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    docs = create_document_from_webpage(url)

    index =  Pinecone.from_documents(docs, embeddings, index_name=st.secrets['PINECONE_NAME']) 

   

    return ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), index.as_retriever(), memory=memory)

def get_agent(url):
    qa = creat_embeddings(url)

    tools = [
        Tool(
            name='Knowledge Base',
            func=qa.run,
            description=(
                'use this tool when answering general knowledge queries to get '
                'more information about the topic'
            )
        )
    ]

    llm = ChatOpenAI(
        openai_api_key=st.secrets['OPENAI_API_KEY'],
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )

    agent = initialize_agent(
        agent='chat-conversational-react-description',
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=3,
        early_stopping_method='generate',
        memory=memory
    )

    return agent


# Text input
txt_input = st.text_area('Enter url', '', height=80)

agent = get_agent(txt_input)
st.set_page_config(page_title="HugChat - An LLM-powered Streamlit app")

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 HugChat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [HugChat](https://github.com/Soulter/hugging-chat-api)
    - [OpenAssistant/oasst-sft-6-llama-30b-xor](https://huggingface.co/OpenAssistant/oasst-sft-6-llama-30b-xor) LLM model
    
    💡 Note: No API key required!
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by [Data Professor](https://youtube.com/dataprofessor)')

# Generate empty lists for generated and past.
## generated stores AI generated responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I'm HugChat, How may I help you?"]
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']  

# Layout of input/response containers
input_container = st.container()
colored_header(label='', description='', color_name='blue-30')
response_container = st.container()

# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text
## Applying the user input box
with input_container:
    user_input = get_text()

# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(prompt):
    chatbot = agent
    response = chatbot.chat(prompt)
    return response    


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