#--------------------------------- Import packages ---------------------------------

from flask import Flask, request, jsonify
from dotenv import load_dotenv
from twilio.rest import Client
from langchain.agents import initialize_agent
import main_functions    
#-----------------------------------------------------------------------------------

#---------------------------------- Flask app --------------------------------------
app = Flask(__name__)
client = Client()


@app.route('/chatbot', methods=['POST'])
def chatbot():
    return main_functions.chatbot()

if __name__ == '__main__':
    app.run(debug=False, port=5002)

#------------------------------------------------------------------------------------