import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import itertools
from glob import glob
from PIL import Image

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization

base_find_crack = './SDNET2018/D/CD'
base_find_wall = './SDNET2018/D/UD'
base_coarse_crack = './SDNET2018/P/CP'
base_coarse_wall = './SDNET2018/P/UP'
base_integrate_crack = './SDNET2018/W/CW'
base_integrate_wall = './SDNET2018/W/UW'


find_crack_df = pd.DataFrame(columns=( 'path', 'image', 'label'), index=np.arange(0, 2025))
find_wall_df = pd.DataFrame(columns=( 'path', 'image', 'label'), index=np.arange(0, 11595))
coarse_crack_df = pd.DataFrame(columns=( 'path', 'image', 'label'), index=np.arange(0, 2608))
coarse_wall_df = pd.DataFrame(columns=( 'path', 'image', 'label'), index=np.arange(0, 21726))
integrate_crack_df = pd.DataFrame(columns=( 'path', 'image', 'label'), index=np.arange(0, 3851))
integrate_wall_df = pd.DataFrame(columns=( 'path', 'image', 'label'), index=np.arange(0, 14287))

imageid_path_dict_find_crack = np.array([x for x in glob(os.path.join(base_find_crack, '*.jpg'))])
imageid_path_dict_find_wall = np.array([x for x in glob(os.path.join(base_find_wall, '*.jpg'))])
imageid_path_dict_coarse_crack = np.array([x for x in glob(os.path.join(base_coarse_crack, '*.jpg'))])
imageid_path_dict_coarse_wall = np.array([x for x in glob(os.path.join(base_coarse_wall, '*.jpg'))])
imageid_path_dict_integrate_crack = np.array([x for x in glob(os.path.join(base_integrate_crack, '*.jpg'))])
imageid_path_dict_integrate_wall = np.array([x for x in glob(os.path.join(base_integrate_wall, '*.jpg'))])


find_crack_df['path'] = imageid_path_dict_find_crack
find_crack_df['label'] = 0
find_crack_df['image'] = find_crack_df['path'].map(lambda x: np.asarray(Image.open(x).resize((75, 100))))

find_wall_df['path'] = imageid_path_dict_find_wall
find_crack_df['label'] = 1
find_crack_df['image'] = find_wall_df['path'].map(lambda x: np.asarray(Image.open(x).resize((75, 100))))

coarse_crack_df['path'] = imageid_path_dict_coarse_crack
coarse_crack_df['label'] = 2
coarse_crack_df['image'] = coarse_crack_df['path'].map(lambda x: np.asarray(Image.open(x).resize((75, 100))))

coarse_wall_df['path'] = imageid_path_dict_coarse_wall
coarse_wall_df['label'] = 3
coarse_wall_df['image'] = coarse_wall_df['path'].map(lambda x: np.asarray(Image.open(x).resize((75, 100))))

integrate_crack_df['path'] = imageid_path_dict_integrate_crack
integrate_crack_df['label'] = 4
integrate_crack_df['image'] = integrate_crack_df['path'].map(lambda x: np.asarray(Image.open(x).resize((75, 100))))

integrate_wall_df['path'] = imageid_path_dict_integrate_wall
integrate_wall_df['label'] = 5
integrate_wall_df['image'] = integrate_wall_df['path'].map(lambda x: np.asarray(Image.open(x).resize((75, 100))))


print(imageid_path_dict_find_crack.shape)
print(imageid_path_dict_find_wall.shape)
print(imageid_path_dict_coarse_crack.shape)
print(imageid_path_dict_coarse_wall.shape)
print(imageid_path_dict_integrate_crack.shape)
print(imageid_path_dict_integrate_wall.shape)

crack_df = find_crack_df.append(find_wall_df)
crack_df = crack_df.append(coarse_crack_df)
crack_df = crack_df.append(coarse_wall_df)
crack_df = crack_df.append(integrate_crack_df)
crack_df = crack_df.append(integrate_wall_df)

# crack_df.reset_index(drop=True, inplace=True)
print(crack_df.shape)

features = crack_df.drop(columns=['label'], axis=1)
target = crack_df["label"]

x_train_o, x_test_o, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=1234)

print('x train 0 : ', x_train_o)

x_train = np.asarray(x_train_o['image'].tolist())
x_test = np.asarray(x_test_o['image'].tolist())

print('x train : ', x_train.shape)

x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)

x_test_mean = np.mean(x_test)
x_test_std = np.std(x_test)

print('x_mean : ', x_train_mean)
print('x_std : ', x_train_std)
print('x_train : ', x_train)

x_train = (x_train - x_train_mean) / x_train_std
x_test = (x_test - x_test_mean) / x_test_std

HEIGHT = 75
WIDTH = 100

x_train = x_train.reshape(x_train.shape[0], *(HEIGHT, WIDTH, 3))
x_test = x_test.reshape(x_test.shape[0], *(HEIGHT, WIDTH, 3))

# Reshape image in 3 dimensions (height = 100px, width = 100px , canal = 3)
# x_train = np.resize(x_train, (, HEIGHT, WIDTH, 3))
# x_test = np.resize(x_test, ())



# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
input_shape = (HEIGHT, WIDTH, 3)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same', input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same',))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu', padding='Same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='Same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.40))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu',padding = 'Same',input_shape=input_shape))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu',padding = 'Same',))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu',padding = 'Same',input_shape=input_shape))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu',padding = 'Same',))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()

checkpoint = ModelCheckpoint("classification_model_v2_0-SDNET.hdf5",
                             monitor='val_acc',
                             verbose=1,
                             mode='auto',
                             period=1)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,
          y_train,
          validation_split=0.1,
          epochs=50,
          batch_size=256,
          callbacks=[checkpoint])
