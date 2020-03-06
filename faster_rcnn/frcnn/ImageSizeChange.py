import cv2 as cv
import os

OBJ_PATH = './cracks/total'
DES = './cracks/fixed'

if not os.path.exists(DES):
    os.mkdir(DES)


files = os.listdir(OBJ_PATH)

for file in files:
    try:
        image = cv.imread(OBJ_PATH + '/' + file)
        image = cv.resize(image, (512, 512))

        cv.imwrite(DES + '/' + file, image)
    except Exception as e:
        print(e)
