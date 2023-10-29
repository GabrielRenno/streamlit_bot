'''
This is the current up-to-date version of the streamlit whatsapp interface
'''

# ------------------------------------------------ IMPORT  ---------------------------------------------------------- #
import time
import pandas as pd
import pinecone

from datetime import datetime
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import streamlit as st
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Pinecone

import pickle
from pathlib import Path

import streamlit_authenticator as stauth

import yaml
from yaml.loader import SafeLoader
# -----------------------------------------  CREDENTIALS TESTING  ---------------------------------------------------- #
#from credentials import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_API_ENV        # Hashtag out when DEPLOYING
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]                                    # Hashtag out when TESTING
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]                                # Hashtag out when TESTING
PINECONE_API_ENV = st.secrets["PINECONE_API_ENV"]                                # Hashtag out when TESTING


# ------------------------------------------ CONNECT TO VECTORDB  --------------------------------------------------- #
def connect_vectordb():
    # Create Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Initialize Pinecone
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

    vectordb = Pinecone.from_existing_index(index_name="python-index", embedding=embeddings)
    return vectordb


# ------------------------------------------- AUTHENTICATION  ------------------------------------------------------- #
st.set_page_config(page_title="Col-legi Sant Miquel Chatbot", page_icon=":robot_face:")


file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    pickle.load(file)

with open("hashed_pw.pkl", "rb") as file:
    hashed_pw = pickle.load(file)

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login('Login', 'main', )

if authentication_status is False:
    st.error('Username/password is incorrect')

elif authentication_status is None:
    st.warning('Please enter your username and password')

elif authentication_status:
    # ---------------------------------------  APP AFTER AUTHENTICATION  --------------------------------------------- #
    # Variables
    url = f"""https://www.csm.cat/"""
    model = "gpt-4"             #"gpt-3.5-turbo"
    template = """You are a helpful chatbot, named RSLT, designed to assist users with inquiries related to Collegi Sant Miquel, a reputable school in Barcelona. 
    You provide detailed responses based on the information available on the school's official website. 
    Your goal is to engage in a friendly and professional conversation, answering questions, guiding, and providing recommendations to users. 
    However, you must refrain from sharing any information that is not present on the school's website. 
    If a question is deemed unethical, you have the discretion to choose not to answer it. 
    Your responses should always be concise, containing one paragraph or less.
    
    Context: {context}
    Question: {question}
    Helpful Answer:"""


    # -------------------------------------------- AGENT INSTANCE  ------------------------------------------------------ #
    def create_agent(vectordb, model, template):
        vectordb = vectordb

        # Create ChatOpenAI instance
        llm = ChatOpenAI(model_name=model, temperature=0, openai_api_key=OPENAI_API_KEY)

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


    # ----------------------------------------------- RUN AGENT  -------------------------------------------------------- #
    def run_agent(agent, question):
        # Run agent
        result = agent({"question": question})

        return result["answer"]


    # -------------------------------------------- CONVERSATION LOG  ---------------------------------------------------- #
    # Load the conversation log
    conversation_log_file = 'conversation_log.csv'

    try:
        conversation_log = pd.read_csv(conversation_log_file)
    except FileNotFoundError:
        conversation_log = pd.DataFrame(columns=['email', 'question', 'question_time', 'answer', 'answer_time', 'feedback'])

    # Check for duplicate conversations
    def is_duplicate_conversation(username, question, answer):
        similar_conversations = conversation_log[(conversation_log['email'] == username) &
                                                  (conversation_log['question'] == question) &
                                                  (conversation_log['answer'] == answer)]
        return not similar_conversations.empty


    # ----------------------------------------------- MAIN PAGE   ------------------------------------------------------- #
    st.title("Col-legi Sant Miquel Chatbot")
    authenticator.logout('Logout', 'main', key='unique_key')
    with st.expander("Informació en Català"):
        st.markdown("""
        ## Català
        Benvinguts a l'aplicació de proves del Col·legi Sant Miquel Chatbot, impulsada per GPT-4! Aquest chatbot d'última generació està dissenyat per proporcionar informació i respondre preguntes sobre el Col·legi Sant Miquel, una institució reconeguda a Barcelona. GPT-4 utilitza un processament de llenguatge natural avançat per oferir respostes detallades i precises sobre diversos aspectes de l'escola.
        
        Per interactuar amb el chatbot, simplement fes una pregunta o proporciona un tema relacionat amb el Col·legi Sant Miquel a Barcelona. El chatbot utilitzarà la seva extensa base de coneixements i comprensió del tema per generar una resposta completa i informativa. Si us plau, consulta sobre la història de l'escola, programes acadèmics, professorat, instal·lacions del campus, activitats extraescolars, procés d'admissió o qualsevol altre aspecte que vulguis saber més.
        
        Cal tenir en compte que aquesta és una versió de prova i, tot i que el chatbot esforça a oferir informació precisa i útil, potser no sempre disposi de la informació més actualitzada. Agraïm els vostres comentaris i suggeriments mentre seguim millorant i ampliant aquesta aplicació. Feu les vostres preguntes i exploreu el món del Col·legi Sant Miquel amb el nostre chatbot impulsat per GPT-4!
        
        """)

    with st.expander("Information in English"):
        st.markdown("""
        ## English
        Welcome to the Col-legi Sant Miquel Chatbot test App, powered by GPT-4! This cutting-edge chatbot is designed to provide you with information and answer questions about Col-legi Sant Miquel, a renowned institution in Barcelona. GPT-4 leverages advanced natural language processing to deliver detailed and accurate responses regarding various aspects of the school.

        To interact with the chatbot, simply ask a question or provide a topic related to Col-legi Sant Miquel in Barcelona. The chatbot will then utilize its extensive knowledge base and understanding of the topic to generate a comprehensive and informative response. Feel free to inquire about the school's history, academic programs, faculty, campus facilities, extracurricular activities, admissions process, or any other aspects you'd like to know more about.

        Please keep in mind that this is a test version, and while the chatbot strives to offer accurate and helpful information, it may not always have the most up-to-date details. We appreciate your feedback and input as we continue to enhance and improve this app. Ask away and explore the world of Col-legi Sant Miquel with our GPT-4-powered chatbot!
        """)


    # With this code to display a larger text input for the question
    st.markdown("""
    ## **Hola! Com puc ajudar-te amb informació sobre el Col·legi Sant Miquel avui?**
    Fes-me qualsevol pregunta a la caixa de sota. I understand English, Catalan, Spanish, Portuguese, German and Dutch.
    """, unsafe_allow_html=True)

    question = st.text_area(" ... ", key='question_input', height=100, max_chars=500)

    asked_button = st.button("Preguntar 🤖", use_container_width=True)

    if 'vector_db' not in st.session_state:
        st.session_state['vector_db'] = connect_vectordb()

    if 'chain' not in st.session_state:
        st.session_state["chain"] = create_agent(st.session_state["vector_db"], model, template)

    if asked_button:
        question_time = datetime.utcnow()
        answer = run_agent(st.session_state["chain"], question)
        answer_time = datetime.utcnow()


        # Store the conversation in the log
        if not is_duplicate_conversation(username, question, answer):
            conversation_log.loc[len(conversation_log)] = [username, question, question_time, answer, answer_time, 0]


    # ------------------------------------------  CONVERSATION LOG  ------------------------------------------------- #
    #st.markdown("---")  # Add a visual separator
    #st.write("*Your conversation Log:*")

    # Reverse the order of the conversation log
    st.session_state["conversation_log"] = conversation_log[conversation_log['email'] == username].iloc[::-1]

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


    for index, row in st.session_state["conversation_log"].iterrows():

        # Feedback buttons
        col1, col2, col3 = st.columns([12, 1, 1])

        with col1:
            st.markdown(
                f"<div class='conversation-log'><span class='bot-message'>Chatbot:</span> {row['answer']}</div>",
                unsafe_allow_html=True)

            st.markdown(
                f"<div class='conversation-log'><span class='user-message'>You:</span> {row['question']}</div>",
                unsafe_allow_html=True)

            rounded_question_time = str(row['question_time'])
            st.caption(f"{rounded_question_time.split('.')[0]}")


        with col2:
            dislike = st.button(f"👎", key=f"dislike_{index}", disabled=row['feedback'] == -1)
            if dislike:
                # -1 to the feedback column
                conversation_log.loc[index, 'feedback'] = -1
                st.error(" ")


        with col3:
            like = st.button(f"👍", key=f"like_{index}", disabled=row['feedback'] == 1)
            if like:
                # +1 to the feedback column
                conversation_log.loc[index, 'feedback'] = 1
                st.success(" ")

        if dislike or like:
            st.toast("Thank you for the feedback!  🙏")
            st.balloons()



    # Save conversation log as a csv file
    conversation_log.to_csv(conversation_log_file, index=False)

