import base64
import io
from PIL import Image
import numpy as np
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import mediapipe as mp
from keras.models import load_model
import time
import pandas as pd


def readb64(base64_string):
    idx = base64_string.find('base64,')
    base64_string = base64_string[idx + 7:]

    sbuf = io.BytesIO()

    sbuf.write(base64.b64decode(base64_string, ' /'))
    pimg = Image.open(sbuf)

    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)


def readAndSaveImg(data_image):
    idx = data_image.find('base64,')
    base64_string = data_image[idx + 7:]
    img = base64.b64decode(base64_string)
    dir_list = os.listdir("./img")
    next_filename = f"./img/{len(dir_list)}.png"
    with open(next_filename, "wb") as f:
        f.write(img)
    # Using cv2.imread() method
    # img = cv2.imread(next_filename)
    # predict_img(img)

def testFunction():
    dir_list = os.listdir("./img")
    length = len(dir_list) - 1
    for x in range(1, length):
        next_filename = f"./img/{x}.png"
        img = cv2.imread(next_filename)
        predict_img(img)

def predict_img(data_image):
    model = load_model('smnist.h5')

    mphands = mp.solutions.hands
    hands = mphands.Hands()

    h, w, c = data_image.shape

    letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                  'V', 'W', 'X', 'Y']

    analysisframe = data_image
    cv2.imshow("Frame", data_image)
    framergbanalysis = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2RGB)
    resultanalysis = hands.process(framergbanalysis)
    hand_landmarksanalysis = resultanalysis.multi_hand_landmarks
    if hand_landmarksanalysis:
        for handLMsanalysis in hand_landmarksanalysis:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lmanalysis in handLMsanalysis.landmark:
                x, y = int(lmanalysis.x * w), int(lmanalysis.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            y_min -= 20
            y_max += 20
            x_min -= 20
            x_max += 20
    else:
        print("Hands not found")
        return

    analysisframe = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2GRAY)
    analysisframe = analysisframe[y_min:y_max, x_min:x_max]
    try:
        analysisframe = cv2.resize(analysisframe, (28, 28))
    except:
        return

    nlist = []
    rows, cols = analysisframe.shape
    for i in range(rows):
        for j in range(cols):
            k = analysisframe[i, j]
            nlist.append(k)

    datan = pd.DataFrame(nlist).T
    colname = []
    for val in range(784):
        colname.append(val)
    datan.columns = colname

    pixeldata = datan.values
    pixeldata = pixeldata / 255
    pixeldata = pixeldata.reshape(-1, 28, 28, 1)
    prediction = model.predict(pixeldata)
    predarray = np.array(prediction[0])
    letter_prediction_dict = {letterpred[i]: predarray[i] for i in range(len(letterpred))}
    predarrayordered = sorted(predarray, reverse=True)
    high1 = predarrayordered[0]
    high2 = predarrayordered[1]
    high3 = predarrayordered[2]
    for key, value in letter_prediction_dict.items():
        if value == high1:
            print("Predicted Character 1: ", key)
            print('Confidence 1: ', 100 * value)
        elif value == high2:
            print("Predicted Character 2: ", key)
            print('Confidence 2: ', 100 * value)
        elif value == high3:
            print("Predicted Character 3: ", key)
            print('Confidence 3: ', 100 * value)
    framergb = cv2.cvtColor(data_image, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            y_min -= 20
            y_max += 20
            x_min -= 20
            x_max += 20
            cv2.rectangle(data_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.imshow("Frame", data_image)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    testFunction()