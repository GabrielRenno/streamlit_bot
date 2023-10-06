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
    st.markdown("""
## Catalan
Benvinguts a l'aplicació de proves del Col·legi Sant Miquel Chatbot, impulsada per GPT-4! Aquest chatbot d'última generació està dissenyat per proporcionar informació i respondre preguntes sobre el Col·legi Sant Miquel, una institució reconeguda a Barcelona. GPT-4 utilitza un processament de llenguatge natural avançat per oferir respostes detallades i precises sobre diversos aspectes de l'escola.

Per interactuar amb el chatbot, simplement fes una pregunta o proporciona un tema relacionat amb el Col·legi Sant Miquel a Barcelona. El chatbot utilitzarà la seva extensa base de coneixements i comprensió del tema per generar una resposta completa i informativa. Si us plau, consulta sobre la història de l'escola, programes acadèmics, professorat, instal·lacions del campus, activitats extraescolars, procés d'admissió o qualsevol altre aspecte que vulguis saber més.

Cal tenir en compte que aquesta és una versió de prova i, tot i que el chatbot esforça a oferir informació precisa i útil, potser no sempre disposi de la informació més actualitzada. Agraïm els vostres comentaris i suggeriments mentre seguim millorant i ampliant aquesta aplicació. Feu les vostres preguntes i exploreu el món del Col·legi Sant Miquel amb el nostre chatbot impulsat per GPT-4!

## English
Welcome to the Col-legi Sant Miquel Chatbot test App, powered by GPT-4! This cutting-edge chatbot is designed to provide you with information and answer questions about Col-legi Sant Miquel, a renowned institution in Barcelona. GPT-4 leverages advanced natural language processing to deliver detailed and accurate responses regarding various aspects of the school.

To interact with the chatbot, simply ask a question or provide a topic related to Col-legi Sant Miquel in Barcelona. The chatbot will then utilize its extensive knowledge base and understanding of the topic to generate a comprehensive and informative response. Feel free to inquire about the school's history, academic programs, faculty, campus facilities, extracurricular activities, admissions process, or any other aspects you'd like to know more about.

Please keep in mind that this is a test version, and while the chatbot strives to offer accurate and helpful information, it may not always have the most up-to-date details. We appreciate your feedback and input as we continue to enhance and improve this app. Ask away and explore the world of Col-legi Sant Miquel with our GPT-4-powered chatbot!
""")


    # With this code to display a larger text input for the question
    st.markdown("""
## **Hello! How can I assist you with information about Col·legi Sant Miquel today?**
Ask me anything in the box below.
""", unsafe_allow_html=True)

    question = st.text_area("", key='question_input', height=100, max_chars=500)

    if st.button("Ask"):
        answer = run_agent(agent, question)
        #st.write("***You:***", question)
        #st.write("***Chatbot:***", answer)

        if not is_duplicate_conversation(email, question, answer):
            conversation_log.loc[len(conversation_log)] = [email, question, answer, datetime.utcnow()]

    #st.markdown("---")  # Add a visual separator
    #st.write("*Your conversation Log:*")
    
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














