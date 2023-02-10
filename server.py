import os
import socket
from flask import Flask, make_response, request
from flask_socketio import SocketIO, emit
from image_processing import readAndSaveImg, predict_img
import eventlet
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

strSentence = ""

@socketio.on('image')
def image(data_image):
    global strSentence
    global letters
    imgFileName = readAndSaveImg(data_image)
    prediction = predict_img(imgFileName)
    if prediction != " ":
         strSentence += prediction
    emit('response_back', [prediction])

@socketio.on('text')
def autocorrect(text_data):
    spell = Speller()
    corrected = spell(text_data)
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
    ip = socket.gethostbyname("localhost")
    socketio.run(app,host=ip,port=9990 ,debug=True)