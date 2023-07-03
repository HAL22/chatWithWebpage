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
        openai_api_key="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",
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

