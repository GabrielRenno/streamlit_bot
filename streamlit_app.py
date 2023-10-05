
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
# Pip install tiktoken
!pip install tiktoken
from twilio.rest import Client
import requests
import time
from langchain.agents import initialize_agent
import pandas as pd
from dotenv import load_dotenv
from bs4 import BeautifulSoup


openai.api_key = st.secrets["OPENAI_API_KEY"]

# -------------------------------------------------------------------------------------------
# --------------------------------------- Import Keys ---------------------------------------
# Load environment variables

#base_url = "https://api.assemblyai.com/v2"

#headers = {
   # "authorization": #ASSEMBLYAI_API_KEY
#}

#load_dotenv()

# -------------------------------------------------------------------------------------------
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
    embeddings = OpenAIEmbeddings()

    # Create Vector Database
    vectordb = FAISS.from_texts(texts=splits, embedding=embeddings)

    return vectordb


# --------------------------------------------------------------------------------------------
# --------------------------------------- Create Agent ---------------------------------------
def create_agent(vectordb, model, template):
    vectordb = vectordb

    # Create ChatOpenAI instance
    llm = ChatOpenAI(model_name=model, temperature=0)

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
# --------------------------------------- Voicenote to text -----------------------------------------

#def voicenote(upload_url):
    data = {
        "audio_url": upload_url  # You can also use a URL to an audio or video file on the web
    }
    url = base_url + "/transcript"
    response = requests.post(url, json=data, headers=headers)

    transcript_id = response.json()['id']
    polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"

    while True:
        transcription_result = requests.get(polling_endpoint, headers=headers).json()

        if transcription_result['status'] == 'completed':
            return transcription_result['text']

        elif transcription_result['status'] == 'error':
            raise RuntimeError(f"Transcription failed: {transcription_result['error']}")

        else:
            time.sleep(3)


# ---------------------------------------------------------------------------------------------------

# ---------------------------------------- Data Collection ------------------------------------------

#def data_collection(answer):
    num_media = request.values.get('NumMedia', '')
    sms_id = request.values.get('SmsSid', '')
    wa_id = request.values.get('WaId', '')
    body = request.values.get('Body', '')
    timestamp = datetime.datetime.now()
    answer = answer.strip()

    # store into a dataframe
    # Sample data to append
    data_to_append = {
        'NumMedia': [num_media],
        'SmsSid': [sms_id],
        'WaId': [wa_id],
        'Body': [body],
        'Answer': [answer],
        'Timestamp': [timestamp]
    }

    # Create a new DataFrame from the data
    new_data_df = pd.DataFrame(data_to_append)

    # Read the existing DataFrame from the CSV file
    existing_df = pd.read_csv("Data_Lake/messages.csv")

    # Concatenate the existing DataFrame and the new DataFrame
    df = pd.concat([existing_df, new_data_df], ignore_index=True)

    # Save the combined DataFrame to a CSV file
    df.to_csv("Data_Lake/messages.csv", index=False)

# ---------------------------------------------------------------------------------------------------


url = f"""https://www.csm.cat/"""
vector_db = create_vectordb([url])
model = "gpt-3.5-turbo"
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

user_data = {
    'ruben.tak@rslt.agency': 'ruben',
    'gabriel.renno@rslt.agency': 'gabriel',
    'nils.jennissen@rslt.agency': 'nils',
    'onassis.nottage@rslt.agency': 'onassis',
}

conversation_log_file = 'conversation_log.csv'
try:
    conversation_log = pd.read_csv(conversation_log_file)
except FileNotFoundError:
    conversation_log = pd.DataFrame(columns=['Email', 'User Message', 'System Answer', 'Time'])

def authenticate_user(email, password):
    if email in user_data:
        return user_data[email] == password
    return False

def is_duplicate_conversation(email, question, answer):
    similar_conversations = conversation_log[(conversation_log['Email'] == email) &
                                              (conversation_log['User Message'] == question) &
                                              (conversation_log['System Answer'] == answer)]
    return not similar_conversations.empty


def display_main_page(email):
    st.title("Col-legi Sant Miquel Chatbot")
    st.write("Welcome to the Col-legi Sant Miquel Chatbot test App. Ask a question and the chatbot will reply. The chatbot uses GPT-4 to answer questions about the Col-legi Sant Miquel in Barcelona. This is the first version in test.")
    #st.image("/Users/gabrielrenno/Documents/wpp_chatbot/WhatsAppBot/image.png", use_column_width=True)

    question = st.text_input("Ask a question:", key='question_input')

    # Status logged in

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



# if 'is_logged_in' not in st.session_state:
   #  st.session_state['is_logged_in'] = False

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

elif 'email' in st.session_state:  # Check if 'email' is in st.session_state before trying to access it
    display_main_page(st.session_state['email'])

else:
    st.error("Please log in to continue.")

conversation_log.to_csv(conversation_log_file, index=False)




