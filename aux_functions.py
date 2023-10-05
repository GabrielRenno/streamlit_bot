#----------------------------------------- Import packages -----------------------------------------
from flask import Flask, request, jsonify
from twilio.twiml.messaging_response import MessagingResponse
from credentials import OPENAI_API_KEY, ASSEMBLYAI_API_KEY
from dotenv import load_dotenv
from twilio.rest import Client
import requests
import time
from langchain.agents import initialize_agent
from credentials import OPENAI_API_KEY
import pandas as pd
import datetime
from dotenv import load_dotenv
#---------------------------------------------------------------------------------------------------

#-------------------------------------- Import Keys ------------------------------------------------
base_url = "https://api.assemblyai.com/v2"

headers = {
    "authorization": ASSEMBLYAI_API_KEY 
}

load_dotenv()
#---------------------------------------------------------------------------------------------------


#--------------------------------------- Voicenote to text -----------------------------------------

def voicenote(upload_url):    
  data = {
    "audio_url": upload_url # You can also use a URL to an audio or video file on the web
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
#---------------------------------------------------------------------------------------------------

#---------------------------------------- Data Collection ------------------------------------------

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
  
 #---------------------------------------------------------------------------------------------------


