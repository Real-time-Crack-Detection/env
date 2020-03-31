import numpy as np
import cv2 as cv
import os

PATH = './cracks/total'
cracks = os.listdir(PATH)

heights = []
widths = []

for file in cracks:
    print(PATH + '/' + file)
    try:
        image = cv.imread(PATH + '/' + file)
        height, width, channel = image.shape
        heights.append(height)
        widths.append(width)
    except Exception as e:
        print(e)


sum_w = np.sum(heights)
sum_h = np.sum(widths)

print('total num : ', len(heights))
print('total w :', sum_w)
print('total h :', sum_h)
print('avg width : ', sum_w // len(widths))
print('avg height : ', sum_h // len(heights))