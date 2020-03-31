

######################### File preprocessing part #############################
import shutil
import os

POSITIVE_PATH = './Dataset/surface-crack-detection/Positive'
NEGATIVE_PATH = './Dataset/surface-crack-detection/Negative'
TOTAL_PATH = './Dataset/surface-crack-detection/Total'
TRAIN_PATH = './Dataset/surface-crack-detection/Train'
TEST_PATH = './Dataset/surface-crack-detection/Test'

if not os.path.exists(POSITIVE_PATH):
    os.mkdir(POSITIVE_PATH)
if not os.path.exists(NEGATIVE_PATH):
    os.mkdir(NEGATIVE_PATH)
if not os.path.exists(TOTAL_PATH):
    os.mkdir(TOTAL_PATH)
if not os.path.exists(TRAIN_PATH):
    os.mkdir(TOTAL_PATH)
if not os.path.exists(TEST_PATH):
    os.mkdir(TOTAL_PATH)

#### naming of Non-crack ####
def file_name_change():
    positives = os.listdir(POSITIVE_PATH)
    for p in positives:
        os.rename(POSITIVE_PATH + '/' + p, POSITIVE_PATH + '/positive' + p)
        ## before = 00001.jpg / after = positive00001.jpg

    #### naming of Crack ####
    negatives = os.listdir(NEGATIVE_PATH)
    for n in negatives:
        os.rename(NEGATIVE_PATH + '/' + n, NEGATIVE_PATH + '/negative' + n)
        ## before = 00001.jpg / after = negative00001.jpg

#### All file move to 'Total' ####
def move_file_to_all():
    positives = os.listdir(POSITIVE_PATH)
    negatives = os.listdir(NEGATIVE_PATH)

    for p in positives:
        shutil.copy(POSITIVE_PATH + '/' + p, TOTAL_PATH + '/' + p)
    for n in negatives:
        shutil.copy(NEGATIVE_PATH + '/' + n, TOTAL_PATH + '/' + n)

#### This function move Certain file from 'src' to 'des' until 0 ~ 'limit'
def move_file_to_path(src, des, limit=0):
    list_image = os.listdir(src)
    count = 1
    for img in list_image:
        if count > limit:
            break
        count += 1

        shutil.copy(src + '/' + img, des + '/' + img)

#move_file_to_path(POSITIVE_PATH, TRAIN_PATH, limit=6000)
#move_file_to_path(NEGATIVE_PATH, TRAIN_PATH, limit=6000)
#move_file_to_path(POSITIVE_PATH, TEST_PATH, limit=3000)
#move_file_to_path(NEGATIVE_PATH, TEST_PATH, limit=3000)
##############################################################################

############################ Data making part ################################
import numpy as np
import cv2 as cv
import random

IMAGE_SIZE = 112

def image_load(path, IMAGE_SIZE, save_name = '', suffle=False):
    list_x = os.listdir(path)
    image_x = []
    total = len(list_x)
    i = 0

    print(list_x)

    if suffle:
        random.seed(1)
        random.shuffle(list_x)
    # If user want shuffle the data, give "shuffle=True"

    for x in list_x:
        try:
            image = cv.resize(cv.imread(path + '/' + x, cv.IMREAD_ANYCOLOR),
                               (IMAGE_SIZE, IMAGE_SIZE)) / 255
            image_x.append(image)

            i += 1
            if i % 100 == 0:
                print(i, ' / ', len(list_x))
        except Exception as e:
            print(x)
    if save_name is not '':
        np.save(save_name, image_x)
    return image_x, list_x

def image_twoPath_load(path1, path2, IMAGE_SIZE, save_name = '', thresh = 0, suffle=False):
    image_x = []

    list_x1 = os.listdir(path1)
    list_x2 = os.listdir(path2)

    if suffle:
        random.seed(1)
        random.shuffle(list_x1)
        random.shuffle(list_x2)
    # If user want shuffle the data, give "shuffle=True"
    print('Start imagezation.')
    for x in list_x1[thresh-100:thresh]:
        try:
            image = cv.resize(cv.imread(path1 + '/' + x, cv.IMREAD_ANYCOLOR),
                               (IMAGE_SIZE, IMAGE_SIZE)) / 255
            image_x.append(image)
        except Exception as e:
            print(x)

    for x in list_x2[thresh-100:thresh]:
        try:
            image = cv.resize(cv.imread(path2 + '/' + x, cv.IMREAD_ANYCOLOR),
                               (IMAGE_SIZE, IMAGE_SIZE)) / 255
            image_x.append(image)
        except Exception as e:
            print(x)

    print('Saving...')
    if save_name is not '':
        np.save(save_name, image_x)
    print('Done.')
    return image_x


### 아직 라벨 만드는 건 익숙치 않아서, 하드 코딩으로 하도록 함.. ###
def label_make(image_x, threshold=None):
    # threshold는 임계값으로, po - ne가 몇 번째부터 나누어지는지를 의미함.
    # ex. po 3000개, ne 3000개 => 3001번째부터 나누어지므로, threshold는 3000임.
    labels = []
    if threshold is None:
        print('---- It is impossible because no input threshold. ----')
        return None

    count = 0
    for i in range (0, len(image_x)):
        if count < threshold:
            labels.append([1., 0.]) # Non-crack
        else:
            labels.append([0., 1.]) # Crack
        count += 1
    labels = np.asarray(labels, dtype=np.float32)
    return labels

# move_file_to_path(POSITIVE_PATH, TRAIN_PATH, limit=15000)
# move_file_to_path(NEGATIVE_PATH, TRAIN_PATH, limit=15000)
#TEMP_PATH = './Dataset/surface-crack-detection/temp'
#image_load(TEMP_PATH, IMAGE_SIZE, 'test_train.npy', False)
#image_load(TEMP_PATH, IMAGE_SIZE, 'test_test.npy', False)

# list_image = os.listdir(NEGATIVE_PATH)
# count = 1
# for img in list_image[15000:]:
#     shutil.copy(NEGATIVE_PATH + '/' + img, TEST_PATH + '/' + img)
#
# list_image = os.listdir(POSITIVE_PATH)
# count = 1
# for img in list_image[15000:]:
#     shutil.copy(POSITIVE_PATH + '/' + img, TEST_PATH + '/' + img)\
#image_twoPath_load(NEGATIVE_PATH, POSITIVE_PATH, IMAGE_SIZE, 'test_train.npy', thresh=100)
#image_twoPath_load(NEGATIVE_PATH, POSITIVE_PATH, IMAGE_SIZE, 'test_test.npy', thresh=100)
# image_twoPath_load(NEGATIVE_PATH, POSITIVE_PATH, IMAGE_SIZE, 'Train3.npy', thresh=15000)
# image_twoPath_load(NEGATIVE_PATH, POSITIVE_PATH, IMAGE_SIZE, 'Train4.npy', thresh=20000)

#image_load(TEST_PATH, IMAGE_SIZE, 'Test.npy')