import base64
import io
from PIL import Image
import numpy as np
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import mediapipe as mp
from keras.models import load_model
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
    return next_filename

def predict_img(filename):
    img = cv2.imread(filename)
    model = load_model('smnist.h5')

    mphands = mp.solutions.hands
    hands = mphands.Hands()

    height, width, c = img.shape

    PredictionLetters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                  'V', 'W', 'X', 'Y']

    FrameToBeAnalysed = img
    FrameRGBToBeAnalysed = cv2.cvtColor(FrameToBeAnalysed, cv2.COLOR_BGR2RGB)
    ResultOfAnalysis = hands.process(FrameRGBToBeAnalysed)
    HandAnalysis = ResultOfAnalysis.multi_hand_landmarks
    if HandAnalysis:
        for HandLandMarkAnalysis in HandAnalysis:
            x_max = 0
            y_max = 0
            x_min = width
            y_min = height
            for LandMarkAnalysis in HandLandMarkAnalysis.landmark:
                x, y = int(LandMarkAnalysis.x * width), int(LandMarkAnalysis.y * height)
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
        return " "

    FrameToBeAnalysed = cv2.cvtColor(FrameToBeAnalysed, cv2.COLOR_BGR2GRAY)
    FrameToBeAnalysed = FrameToBeAnalysed[y_min:y_max, x_min:x_max]
    try:
        FrameToBeAnalysed = cv2.resize(FrameToBeAnalysed, (28, 28))
    except:
        return " "

    nlist = []
    rows, cols = FrameToBeAnalysed.shape
    for i in range(rows):
        for j in range(cols):
            k = FrameToBeAnalysed[i, j]
            nlist.append(k)

    datan = pd.DataFrame(nlist).T
    colname = []
    for val in range(784):
        colname.append(val)
    datan.columns = colname

    PixelData = datan.values
    PixelData = PixelData / 255
    PixelData = PixelData.reshape(-1, 28, 28, 1)
    PredictionFromModel = model.predict(PixelData)
    PredictionArray = np.array(PredictionFromModel[0])
    LetterPredictionDictionary = {PredictionLetters[i]: PredictionArray[i] for i in range(len(PredictionLetters))}
    SortedPredictionArray = sorted(PredictionArray, reverse=True)
    high1 = SortedPredictionArray[0]
    LetterPrediction = None
    for key, value in LetterPredictionDictionary.items():
        if value == high1:
            LetterPrediction = key
    FrameRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(FrameRGB)
    HandLandMark = result.multi_hand_landmarks
    if HandLandMark:
        for HandLandMark2 in HandLandMark:
            x_max = 0
            y_max = 0
            x_min = width
            y_min = height
            for lm in HandLandMark2.landmark:
                x, y = int(lm.x * width), int(lm.y * height)
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
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.destroyAllWindows()
    return LetterPrediction
