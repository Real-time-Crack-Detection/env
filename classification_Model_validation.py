from keras.models import load_model

MODEL_PATH = 'classification_model_v1_0_kaggle.hdf5'

model = load_model(MODEL_PATH)



import os
import cv2
import numpy as np

TEST_PATH1 = './cracks/Negative'
TEST_PATH2 = './cracks/Positive'

images1 = []
images2 = []

files = os.listdir(TEST_PATH1)
for f in files[0:100]:
    image = cv2.imread(TEST_PATH1 + '/' + f)
    image = cv2.resize(image, (100, 75))

    images1.append(image)

images1 = np.array(images1)

result = model.predict_classes(images1)

print('first')
for r in result:
    print(r)

files = os.listdir(TEST_PATH2)
for f in files[0:100]:
    image = cv2.imread(TEST_PATH2 + '/' + f)
    image = cv2.resize(image, (100, 75))

    images2.append(image)

images2 = np.array(images2)

result = model.predict_classes(images2)

print('second')
for r in result:
    print(r)