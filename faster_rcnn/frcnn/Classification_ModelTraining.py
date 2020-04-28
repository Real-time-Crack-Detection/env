import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import DetectionMethod as Method
import os
import cv2 as cv
import random
## 앞의 API를 이용하여, npy를 읽어들이고, label를 읽어들이고 train 시키면 됨 ##

print('Data loading....')
# 실제 학습 시킬 코드

CD_PATH = './SDNET2018/D/CD'
UD_PATH = './SDNET2018/D/UD'
CP_PATH = './SDNET2018/P/CP'
UP_PATH = './SDNET2018/P/UP'
CW_PATH = './SDNET2018/W/CW'
UW_PATH = './SDNET2018/W/UW'

batch_size = 5000

def get_one_hot_encoding(class_num):
    arr = [[1., 0., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0., 0.],
           [0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 1., 0., 0.],
           [0., 0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 0., 1.]]

    return arr[class_num]

def get_all_path():
    texts = []
    labels = []

    datas = os.listdir(CD_PATH)
    for d in datas:
        texts.append(CD_PATH + '/' + d)
        labels.append(0)

    datas = os.listdir(UD_PATH)
    for d in datas:
        texts.append(UD_PATH + '/' + d)
        labels.append(1)

    datas = os.listdir(CP_PATH)
    for d in datas:
        texts.append(CP_PATH + '/' + d)
        labels.append(2)

    datas = os.listdir(UP_PATH)
    for d in datas:
        texts.append(UP_PATH + '/' + d)
        labels.append(3)

    datas = os.listdir(CW_PATH)
    for d in datas:
        texts.append(CW_PATH + '/' + d)
        labels.append(4)

    datas = os.listdir(UW_PATH)
    for d in datas:
        texts.append(UW_PATH + '/' + d)
        labels.append(5)

    return texts, labels


print('Modeling...')
model = Sequential()
model.add(Conv2D(input_shape=(256, 256, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2),strides=(2, 2)))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2),strides=(2, 2)))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=4096, activation="relu"))
model.add(Dense(units=1024, activation="relu"))
model.add(Dense(units=6, activation="softmax"))

# from keras.models import load_model
# model = load_model('./detection_v1_3.hdf5')

from keras.optimizers import Adam
opt = Adam(lr=0.001)
model.compile(optimizer=opt,
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
#model.summary()

checkpoint = ModelCheckpoint("classification_model_v1_0.hdf5",
                             monitor='val_acc',
                             verbose=1,
                             mode='auto',
                             period=1)

# early = EarlyStopping(monitor='val_acc',
#                       min_delta=0,

#                       patience=20,
#                       verbose=1,
#                       mode='auto')

texts, labels = get_all_path()


random.seed(0)
random.shuffle(texts)
random.shuffle(labels)


total_img = len(texts)  # 전체 훈련 이미지 개수
total_batch = int(total_img / batch_size)  # 학습 시 불러야하는 배치의 수


for epoch in range(10):
    print('total epoch : ', epoch)
    for batch_count in range(total_batch):

        print('Current Process = (', total_batch, '/', batch_count, ')')

        images = []  # 이미지가 들어갈 리스트, 다음 배치 때 초기화된다.
        batch_index = batch_count * batch_size  # 배치 순서 계산

        print('batch_index : ', batch_index)

        train_x = []
        train_y = []

        print('Data Mapping...')

        for name in texts[batch_index: (batch_index + batch_size)]:
            # 0~99, 100~199, 200~299, 300~399...
            # 이미지 데이터 목록에서 해당 배치 순서의 이미지 이름을 받아옴
            img = cv.imread(name)

            train_x.append(img)

            with open('./classification_file.csv', 'r') as f:
                while True:
                    line = f.readline()

                    if not line: break

                    if name == line.split(',')[0]:
                        train_y.append(get_one_hot_encoding(int(line.split(',')[1].strip())))
                        break

        train_x = np.array(train_x)
        train_y = np.array(train_y)

        print('Train x : ', train_x.shape)
        print('Train y : ', train_y.shape)

        print('Training start...')
        model.fit(x=train_x,
                  y=train_y,
                  batch_size=8,
                  verbose=1,
                  epochs=5,
                  callbacks=[checkpoint])

    batch_index = 0