import cv2 as cv
import os

OBJ_PATH = './cracks/Positive'
DES = './cracks/Positive192'

if not os.path.exists(DES):
    os.mkdir(DES)


files = os.listdir(OBJ_PATH)

for file in files:
    try:
        image = cv.imread(OBJ_PATH + '/' + file)
        image = cv.resize(image, (192, 192))

        cv.imwrite(DES + '/' + file, image)
    except Exception as e:
        print(e)
