import base64
import io
from PIL import Image
import numpy as np
import cv2
import os
def readb64(base64_string):
    idx = base64_string.find('base64,')
    base64_string  = base64_string[idx+7:]

    sbuf = io.BytesIO()

    sbuf.write(base64.b64decode(base64_string, ' /'))
    pimg = Image.open(sbuf)


    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

def readAndSaveImg(data_image):
    idx = data_image.find('base64,')
    base64_string = data_image[idx + 7:]
    img = base64.b64decode(base64_string)
    dir_list = os.listdir("./img")
    next_filename = f"./img/{len(dir_list) + 1}.png"
    with open(next_filename, "wb") as f:
        f.write(img)