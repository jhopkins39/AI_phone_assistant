 # pip install twilio

 # lightweight web application to accept incoming HTTP requests from Twilio -- I'm going to use flask (like the example), but we can use whatever
 # pip install virtualenv (need it to sandbox simple web app)

 # virtual env set up instructions
 # also follow the instructions here to be able to set up a TWILIO phone number
 # https://www.twilio.com/docs/voice/quickstart/python

"""
 cd {path/to/project}
 virtualenv .
 source bin/activate
 pip install -r requirements.txt



# there is a difference between the whisper model and the whisper api
# whisper model takes too long to run locally, so use the api - have to input full files to the api

"""
 # requirements.txt: 
"""
Flask
twilio
flask-sock 
simple-websocket
pyngrok
openai
openai-whisper
elevenlabs
google-generativeai
"""

 # then create a python script and run the script (code in the demo)
 # it should be running locally

 # to actually recceive calls, we need to make the application online for twilio to direct phone calls - they recommend ngrok, but AWS or Heroku
 # download ngrok binary (call commands with the ./ngrok and make sure in right directory)
 # after application is running: 
 # will get a public url and can use that as the webhook in the twilio configuration
 # add code to flask app to handle receiving a call (code on the twilio walk through)

 # on twilio console, make sure your phone number knows web hook and add the ngrok url it should be at and now it'll work
 # for basic twilio account, the phone calling a twilio number must be pre-approved

 # Jake's thought: after the main functionality, the response and error handling is all in the lang chain prompts
 # app to handle the phone call - run it in the virtual environment

 # https://www.twilio.com/blog/transcribe-phone-calls-text-real-time-twilio-vosk
 # ------------------------------------------------
from flask import Flask, send_from_directory
from twilio.twiml.voice_response import VoiceResponse, Start, Connect

from flask import request
from flask_sock import Sock, ConnectionClosed
from twilio.rest import Client

import json
import os

import io
import audioop
import base64

import openai
import whisper
import numpy as np
from scipy.io.wavfile import write


import google.generativeai as palm

from elevenlabs import clone, generate, play, set_api_key

import requests

eleven_api_key = ''
google_palm_api_key = ''
openai.api_key = ''
twilio_account_sid = ''
twilio_account_auth_token = ''

set_api_key(eleven_api_key) #for elevenlabs for text-to-speech


app = Flask(__name__)
sock = Sock(app)
twilio_client = Client(twilio_account_sid, twilio_account_auth_token)

# global variable to help with my code
# message1 is the text from speech-to-text
# my_callsid is the twilio callsid
message1 = ""
my_callsid = ''



@app.route("/answer", methods=['POST'])
def answer_call():

    resp = VoiceResponse()
    # could replace this with a "resp.play() after generating a nice audio file of a voice"
    resp.say("Thank you for calling! What can I help you with?", voice='Polly.Amy')
    start = Start()
    start.stream(url=f'wss://{request.host}/stream')
    resp.append(start)
    resp.pause(length=30)
    
    """
request information I can get: 
        request.form["CallSid"]
        request.form["AccountSid"]
        request.form["To"]
        request.form["From"]
        request.form["CallStatus"]
        request.form["Direction"]
        request.form["CallerName"]
        # there are others too but these are most relevant
    """
    global my_callsid
    my_callsid = request.form["CallSid"]

    print(f'Incoming call from {request.form["From"]}')
    return str(resp), 200, {'Content-Type': 'text/xml'}


# TODO: fix this method, so that use the stream from the connect noun (instead of start) and I actually stream the data in properly and then stream data out properly as well
@sock.route('/stream')
def stream(ws):

    call_id = my_callsid
    

    """Receive and transcribe audio stream."""
    # rec = vosk.KaldiRecognizer(model, 16000)
    buffer = np.array([], dtype=np.float32)
    print("buffer: ", buffer)
    i = 0
    while True:
        i += 1
        message = ws.receive()

        # print("message: ")
        # print(message)

        packet = json.loads(message)
        if packet['event'] == 'start':
            print('Streaming is starting')
        elif packet['event'] == 'stop':
            print('\nStreaming has stopped')
        elif packet['event'] == 'media':
            # this is where I get the audio data and can do stuff with it
            audio = base64.b64decode(packet['media']['payload']) # extracts base64 payload from json
            audio = audioop.ulaw2lin(audio, 2) # decode mu-law data to 16bit uncompressed
            audio = audioop.ratecv(audio, 2, 1, 8000, 16000, None)[0] # changes it from 8000 sample rate to the 16000 smaple rate for the model
            raw_audio_data = audio

            # Create a BytesIO object to work with audio data in memory
            audio_io = io.BytesIO(raw_audio_data)
            #sf.write(audio_io, raw_audio_data, 16000, format='wav')
            aud_array = np.frombuffer(raw_audio_data, np.int16).flatten().astype(np.float32) / 32768.0 

            buffer = np.concatenate((buffer, aud_array))

            # model = whisper.load_model("base")
            # result = model.transcribe(aud_array)
            # transcribed = result["text"]

            # 900 iterations of the while loop is a good amount to receive enough data, and not be too long to wait
            if i % 900 == 0:

                write("example.wav", 16000, buffer)
                my_file = "example.wav"
                # Export the audio file in MP3 format
                #audio_file.export(my_file, format="wav")
                audio_file = open(my_file, "rb")

                print("whisper api call!")
                whisper_response = openai.Audio.transcribe(
                        file=audio_file,
                        model="whisper-1",
                        language="en"
                        )
                audio_file.close()

                transcribed = whisper_response.text

                print("transcribed: " )
                print(transcribed)
                global message1
                message1 = transcribed

                # ~~~~~~~~~~~~~~~~~~~~~~ this is the key to what I want to do I think!!!!!!!
                call = twilio_client.calls(call_id).update(method='POST', url=f'http://{request.host}/reply')
                # params = {'text': transcribed}
                # resp.redirect("/reply", params)
                return call.to

                # print("about to return :)")
                # return _read_gen_message(transcribed)

# TODO: make it so that a response from user loops back to the audio assistant and back and forht while having a global chat bot
# TODO: use langchain to do the above and improve the prompt
# TODO: a different model besides PaLM would probably have better results
# handles text input from user to generate speech reply and plays it aloud to user
@app.route("/reply", methods=['POST', 'GET'])
def reply():

    palm.configure(api_key=google_palm_api_key)
    models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
    model = models[0].name

    customer_question = message1
    prompt  = "You are an extremely helpful virtual phone assistance. Please answer the customer's question: " + customer_question + \
                " It's okay if you do not know the answer, just try your best."

    print("prompt to be passed in: ", prompt)

    completion = palm.chat(messages=prompt, temperature=.8)
    llm_response = completion.last

    print("llm response: ")
    print(llm_response)

    # TODO: add error handling for if llm response is bad
    audio = generate(text=llm_response, voice="Bella", model='eleven_monolingual_v1')

    with open("output.mp3", 'wb') as f:
        f.write(audio)
    print("file written I believe")
    #play(audio)


    response = VoiceResponse()
    #response.say(llm_response, voice='Polly.Amy')  # response.play('final_audio_data.mp3')
    response.play(f'http://{request.host}/audio/output.mp3')
    response.pause(length=25)

    print("about to return from reply method")
    return str(response), 200, {'Content-Type': 'text/xml'}



# this function provides a link to the audio file generated in the previous one
@app.route('/audio/<filename>')
def get_audio(filename):
    return send_from_directory('', filename)


if __name__ == "__main__":

    from pyngrok import ngrok
    port = 3000 # 5000 does not work on mac because it is used for airdrop
    curr = ngrok.connect(port, bind_tls=True)
    public_url = curr.public_url
    print()
    print()
    print("public_url: ", public_url)
    print(curr)
    number = twilio_client.incoming_phone_numbers.list()[0]
    number.update(voice_url=public_url + '/answer') #update twilio console phhone number with correct link
    print(f'Waiting for calls on {number.phone_number}')

    app.run(debug=True, port=port)













# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PaLM example code snippet
# getting palm model
# google_palm_api_key = "AIzaSyBDb29Hs3R-tsfJyXsqq5P6oGbYH4aMGvs"
# import google.generativeai as palm

# palm.configure(api_key=google_palm_api_key)

# models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
# model = models[0].name
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# bark example code (bark takes too long on my computer)
# using bark to synthesize audio
# print("importing items from bark")
# from bark import generate_audio, preload_models

    # text_input = "Hello, I'm the automated assistant for Natan at Anote, created by Jake Hopkins! How can I help you today?"

    # print("creating the audio array")
    # wav_data_from_bark = generate_audio(text_input)
    # print("making audio into base64")
    # audio_payload = base64(wav_data_from_bark) #we need the bark api to output an audio payload in base64!!!

    # start = VR.Start()
    # print("creating stream")
    # start.stream(event = 'media', sequenceNumber=1, media = {"track": outbound, "chunk": 1, "timestamp": 3, "payload": audio_payload}) # may need to add a streamSID to the stream parameters

    # print("doing response")
    # resp.append(start)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PaLM example code: 
# chatting with palm
# print("preloading bark models - should take a long time")
# preload_models() # big method that's important for bark to work

# customer_question = text from twilio that goes through whisper
# prompt  = "You are an extremely helpful virtual phone assistance. Please answer the customer's question: {customer_question}. \
#             If you do not know the answer, don't lie. Say you don't know the answer"

# completion = palm.chat(
#         messages=prompt,
#         temperature=.8,
#         # The maximum length of the response
#     )

# first_response_str = completion.last
# read this response back out with bark and twilio
# wait for the next response from the user 


# while prompt != 'q':

#     # running the api call with the prompt
#     # prompt should equal the next customer response to that
#     completion = completion.reply(prompt)

    # completion.last = chatbot's repsonse to customer
    # read completion.last to the customer

    # prompt = reading in cusotmer's item from twilio

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~







