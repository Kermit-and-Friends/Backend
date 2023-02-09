import base64
import os

from flask import Flask, make_response, request
from flask_socketio import SocketIO, emit
from image_processing import readAndSaveImg, testFunction
import eventlet
import random
from autocorrect import Speller
eventlet.monkey_patch()
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins=['http://localhost:63342', 'https://www.piesocket.com',"chrome-extension://hlbdchfgfampdligmnnhgbdocgaibdaj"], logger=True, async_moe = "event" )


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

full_prediction = ""
@socketio.on('image')
def image(data_image):
    global full_prediction
    index = random.randint(0, len(alphabets))
    readAndSaveImg(data_image)
    # emit('response_back', [alphabets[index]])
    # prediction = testFunction()
    # emit()

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
    # app.run()
    socketio.run(app,host="127.0.0.1",port=9990 ,debug=True)