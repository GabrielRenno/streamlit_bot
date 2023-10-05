#------------------------------------- Import packages ---------------------------------------
from dotenv import load_dotenv
import os
from flask import Flask, request, jsonify

# OpenAI related imports
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

# LangChain related imports
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import  HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
#-------------------------------------------------------------------------------------------

#--------------------------------------- Import Keys ---------------------------------------
# Load environment variables
load_dotenv()

# Credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
HUGGING_FACE_API_TOKEN = os.getenv("HUGGING_FACE_API_TOKEN")
# Set your OpenAI API key here
openai.api_key = os.getenv("OPENAI_API_KEY")
#-------------------------------------------------------------------------------------------

#--------------------------------------- Create Vectordb -----------------------------------
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
    #embeddings = OpenAIEmbeddings()

    # Create Vector Database
    vectordb = FAISS.from_texts(texts=splits, embedding=embeddings)

    return vectordb

#--------------------------------------------------------------------------------------------

#--------------------------------------- Create Agent ---------------------------------------
def create_agent(vectordb,model,template):
    vectordb = vectordb

    # Create ChatOpenAI instance
    llm = ChatOpenAI(model_name=model, temperature=0)

    # Build prompt
    template = template 
    prompt = PromptTemplate(input_variables=[ "context", "question"], template=template)
    
    # Conversation
    memory = ConversationBufferMemory(
        memory_key = "chat_history",
        human_prefix = "### Input",
        ai_prefix = "### Response",
        output_key = "answer",
        return_messages = True)

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
#--------------------------------------------------------------------------------------------

#--------------------------------------- Run Agent ------------------------------------------
def run_agent(agent,question):
    # Run agent
    result = agent({"question": question})
    return result["answer"]
#--------------------------------------------------------------------------------------------
