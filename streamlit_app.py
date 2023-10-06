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

# Assume this is your user data (for demonstration purposes)
user_data = {"user@example.com": "password123"}

# Assume this is your GPT-4 agent function (for demonstration purposes)
def run_agent(agent, question):
    # Implement your GPT-4 agent logic here
    return "Sample GPT-4 answer for: " + question

# Authentication function
def authenticate_user(email, password):
    if email in user_data:
        return user_data[email] == password
    return False

# Check for duplicate conversations
def is_duplicate_conversation(email, question, answer):
    similar_conversations = conversation_log[(conversation_log['Email'] == email) &
                                              (conversation_log['User Message'] == question) &
                                              (conversation_log['System Answer'] == answer)]
    return not similar_conversations.empty

# Display the main chat page
def display_main_page(email):
    st.title("Col-legi Sant Miquel Chatbot")
    st.write("Welcome to the Col-legi Sant Miquel Chatbot test App. Ask a question, and the chatbot will reply. The chatbot uses GPT-4 to answer questions about Col-legi Sant Miquel in Barcelona. This is the first version in test.")

    question = st.text_input("Ask a question:", key='question_input')

    if st.button("Ask"):
        answer = run_agent(agent, question)
        st.write("***You:***", question)
        st.write("***Chatbot:***", answer)

        if not is_duplicate_conversation(email, question, answer):
            conversation_log.loc[len(conversation_log)] = [email, question, answer, datetime.utcnow()]

    st.write("*Your conversation Log:*")
    current_session_log = conversation_log[conversation_log['Email'] == email]
    for index, row in current_session_log.iterrows():
        st.write(f"*You:* {row['User Message']}")
        st.write(f"*Chatbot:* {row['System Answer']}")

    # Save conversation log as a csv file
    conversation_log.to_csv(conversation_log_file, index=False)

# Streamlit app logic
st.set_page_config(page_title="Col-legi Sant Miquel Chatbot", page_icon=":robot_face:")

# Check if the user is logged in
if 'is_logged_in' not in st.session_state:
    st.session_state['is_logged_in'] = False

if st.session_state['is_logged_in'] == False:
    st.title("Chatbot Login")

    # Improved styling for email and password input fields
    email = st.text_input("Email:", value='', key='email_input', type='email')
    password = st.text_input("Password:", value='', key='password_input', type="password")

    # Style the login button using Streamlit's style argument
    login_button = st.button("Login", help="Login", key='login_button', 
                             format_func=lambda _: '<span style="color:white;">Login</span>',
                             unsafe_allow_html=True,
                             style={'background-color': '#007BFF', 'padding': '10px 20px'})

    if login_button:
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




