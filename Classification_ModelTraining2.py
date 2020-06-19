import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

import cv2
import numpy as np
import os
from glob import glob
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

np.random.seed(1000)
#Instantiate an empty model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Passing it to a Fully Connected layer
model.add(Flatten())
# 1st Fully Connected Layer
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))

# 2nd Fully Connected Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))

# 3rd Fully Connected Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))

# Output Layer
model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

CRACK_PATH = './cracks/Positive'
NON_CRACK_PATH = './cracks/Negative'

def load_image(path):
    files = os.listdir(path)

    images = []

    for f in files:
        image = cv2.imread(path + '/' + f)
        image = cv2.resize(image, (224, 224))
        images.append(image)

    return np.array(images)


print('Image loading...')
cracks = load_image(CRACK_PATH)
non_cracks = load_image(NON_CRACK_PATH)

images = np.concatenate((cracks, non_cracks))
labels = []

print(images.shape)

print('labeling...')
with open('classification_file_kaggle.csv', 'r') as f:

    while True:
        line = f.readline()
        if not line: break

        labels.append(int(line.split(',')[1].strip()))

labels = to_categorical(labels, num_classes=2)

train_x, test_x, train_y, test_y = train_test_split(images, labels, test_size=0.2, random_state=100)

print('train x : ', train_x.shape)
print('test x : ', test_x.shape)

print('label : ', labels[0])

print('start training..')
model.fit(train_x, train_y,
          batch_size=8,
          epochs=20,
          validation_split=0.1,
          verbose=1
          )