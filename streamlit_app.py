import os

# ------------------------------------- Import packages ---------------------------------------
from dotenv import load_dotenv
import os
from datetime import datetime
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import streamlit as st
# LangChain related imports
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

from flask import Flask, request, jsonify

from twilio.rest import Client
import requests
import time
from langchain.agents import initialize_agent
import pandas as pd
from dotenv import load_dotenv
from bs4 import BeautifulSoup


# --------------------------------------- Create Vectordb -----------------------------------
def create_vectordb(url):
    # Load documents
    loader = WebBaseLoader(url)
    docs = loader.load()

    # Split text
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=10,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )
    splits = r_splitter.split_text(docs[0].page_content)

    # Create Embeddings
    #model_name = "sentence-transformers/all-mpnet-base-v2"
    #model_kwargs = {'device': 'cpu'}
    #encode_kwargs = {'normalize_embeddings': False}
    #embeddings = HuggingFaceEmbeddings(
     #   model_name=model_name,
      #  model_kwargs=model_kwargs,
       # encode_kwargs=encode_kwargs,
    #)
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])

    # Create Vector Database
    vectordb = FAISS.from_texts(texts=splits, embedding=embeddings)

    return vectordb


# --------------------------------------------------------------------------------------------
# --------------------------------------- Create Agent ---------------------------------------
def create_agent(vectordb, model, template):
    vectordb = vectordb

    # Create ChatOpenAI instance
    llm = ChatOpenAI(model_name=model, temperature=0,openai_api_key=st.secrets["OPENAI_API_KEY"])

    # Build prompt
    template = template
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)

    # Conversation
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        human_prefix="### Input",
        ai_prefix="### Response",
        output_key="answer",
        return_messages=True)

    # Build chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=False)

    return qa_chain


# --------------------------------------------------------------------------------------------
# --------------------------------------- Run Agent ------------------------------------------
def run_agent(agent, question):
    # Run agent
    result = agent({"question": question})
    return result["answer"]
# --------------------------------------------------------------------------------------------
# --------------------------------------- Define input variables -----------------------------------------

url = f"""https://www.csm.cat/"""
vector_db = create_vectordb([url])
model = "gpt-4" #"gpt-3.5-turbo"
template = """You are a helpful chatbot, named RSLT, designed to assist users with inquiries related to Collegi Sant Miquel, a reputable school in Barcelona. 
You provide detailed responses based on the information available on the school's official website. 
Your goal is to engage in a friendly and professional conversation, answering questions, guiding, and providing recommendations to users. 
However, you must refrain from sharing any information that is not present on the school's website. 
If a question is deemed unethical, you have the discretion to choose not to answer it. 
Your responses should always be concise, containing one paragraph or less.

Context: {context}
Question: {question}
Helpful Answer:"""

agent = create_agent(vector_db, model, template)
#----------------------------------------------------------------------------------------------------

# --------------------------------------- Define user and passwords -----------------------------------------
user_data = {
    'ruben.tak@rslt.agency': 'ruben',
    'gabriel.renno@rslt.agency': 'gabriel',
    'nils.jennissen@rslt.agency': 'nils',
    'onassis.nottage@rslt.agency': 'onassis',
}
#----------------------------------------------------------------------------------------------------

#-------------------------------------------- App --------------------------------------------------
import streamlit as st
import pandas as pd
from datetime import datetime

# Load the conversation log
conversation_log_file = 'conversation_log.csv'
try:
    conversation_log = pd.read_csv(conversation_log_file)
except FileNotFoundError:
    conversation_log = pd.DataFrame(columns=['Email', 'User Message', 'System Answer', 'Time'])

# Authentication function
def authenticate_user(email, password):
    # Implementation for authentication (not provided in the original code)
    pass

# Check for duplicate conversations
def is_duplicate_conversation(email, question, answer):
    # Implementation for checking duplicate conversations (not provided in the original code)
    pass

# Display the main chat page
def display_main_page(email):
    st.title("Col-legi Sant Miquel Chatbot")
    
    # ... Existing code for the chat interface ...

    # Reset conversation button
    if st.button("Reset Conversation"):
        # Clear conversation log
        conversation_log = pd.DataFrame(columns=['Email', 'User Message', 'System Answer', 'Time'])
    
    # Reverse the order of the conversation log
    reversed_log = conversation_log[conversation_log['Email'] == email].iloc[::-1]
    
    # Style for the conversation log
    conversation_style = """
        <style>
        .conversation-log {
            padding: 10px;
            background-color: #f3f3f3;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .user-message {
            font-weight: bold;
            color: #0066ff;
        }
        .bot-message {
            color: #009900;
        }
        </style>
        """

    st.markdown(conversation_style, unsafe_allow_html=True)

    for index, row in reversed_log.iterrows():
        st.markdown(f"<div class='conversation-log'><span class='bot-message'>Chatbot:</span> {row['System Answer']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='conversation-log'><span class='user-message'>You:</span> {row['User Message']}</div>", unsafe_allow_html=True)

    # Save conversation log as a csv file
    conversation_log.to_csv(conversation_log_file, index=False)

# Rest of the code remains the same

# Streamlit app logic
st.set_page_config(page_title="Col-legi Sant Miquel Chatbot", page_icon=":robot_face:")

# Check if the user is logged in
if 'is_logged_in' not in st.session_state:
    st.session_state['is_logged_in'] = False

if st.session_state['is_logged_in'] == False:
    st.title("Chatbot Login")
    email = st.text_input("Email:")
    password = st.text_input("Password:", type="password")

    if st.button("Login"):
        if authenticate_user(email, password):
            st.session_state['is_logged_in'] = True
            st.session_state['email'] = email
            st.empty()
            display_main_page(email)
        else:
            st.error("Invalid email or password. Please try again.")

elif 'email' in st.session_state:
    display_main_page(st.session_state['email'])

else:
    st.error("Please log in to continue.")















