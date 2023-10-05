# ------------------------------------- Import packages ---------------------------------------
from dotenv import load_dotenv
import os
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
import datetime
from dotenv import load_dotenv

# -------------------------------------------------------------------------------------------
# --------------------------------------- Import Keys ---------------------------------------
# Load environment variables
load_dotenv()

# Credentials
ASSEMBLYAI_API_KEY = st.secrets("ASSEMBLYAI_API_KEY")
OPENAI_API_KEY = st.secrets("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

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
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    # embeddings = OpenAIEmbeddings()

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
# -------------------------------------- Import Keys ------------------------------------------------
base_url = "https://api.assemblyai.com/v2"

headers = {
    "authorization": ASSEMBLYAI_API_KEY
}

load_dotenv()


# ---------------------------------------------------------------------------------------------------
# --------------------------------------- Voicenote to text -----------------------------------------

def voicenote(upload_url):
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

def data_collection(answer):
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
