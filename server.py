import base64
import os
import socket
from flask import Flask, make_response, request
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO, emit
from image_processing import readAndSaveImg, testFunction, predict_img
import eventlet
import random
import constants
from autocorrect import Speller
eventlet.monkey_patch()
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*', logger=True, async_moe = "event" )


for filename in os.listdir("./img"):
    filepath = os.path.join(f"./img", filename)
    if filepath.endswith(".png"):
        os.remove(filepath)
@app.route('/', methods=['POST', 'GET'])

@app.route('/')
def hello():
    return "<h1> This is the server for Kermit and Friends </h1>"

# @socketio.on('catch-frame')
# def catch_frame(data):
#
#     emit('response_back', data)
alphabets = ["a", "b", "c", "d", "e", "f",
             "g", "h", "i", "j", "k", "l",
             "m", "n", "o", "p", "q", "r",
             "s", "t", "u", "v", "h", "i",
             "j", "k", "l", "m", "n", "o",
             "p", "q", "r", "s", "t", "u",
             "v", "w", "x", "y", "z", "1",
             "2", "3", "4", "5", "6", "7",
             "8", "9", " "]
sentence = constants.sample1
letters = [char for char in sentence]
strSentence = ""

@socketio.on('image')
@cross_origin()
def image(data_image):
    global strSentence
    imgFileName = readAndSaveImg(data_image)
    prediction = testFunction(imgFileName)
    if prediction != " ":
        strSentence += prediction
    autocorrect(strSentence)
    # emit('response_back', [prediction])
    emit('response_back', [letters[0]])
    letters.pop(0)

@socketio.on('text')
@cross_origin()
def autocorrect(text_data):
    spell = Speller()
    corrected = spell(text_data)
    print(text_data)
    print(corrected)
    emit('autocorrected', [corrected])

def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

if __name__ == '__main__':
    # app.run()
    ip = socket.gethostbyname("localhost")
    print(ip)
    socketio.run(app,host=ip,port=9990 ,debug=True)