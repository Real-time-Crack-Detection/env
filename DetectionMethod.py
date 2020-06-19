

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
    for x in list_x1:
        try:
            image = cv.resize(cv.imread(path1 + '/' + x, cv.IMREAD_ANYCOLOR),
                               (IMAGE_SIZE, IMAGE_SIZE))
            image_x.append(image)
        except Exception as e:
            print(x)

    for x in list_x2:
        try:
            image = cv.resize(cv.imread(path2 + '/' + x, cv.IMREAD_ANYCOLOR),
                               (IMAGE_SIZE, IMAGE_SIZE))
            image_x.append(image)
        except Exception as e:
            print(x)

    print('Saving...')
    if save_name is not '':
        np.save(save_name, image_x)
    print('Done.')
    return image_x


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

import os
import shutil
import numpy as np
import time

FILES_PATH = './data/img'
TRAIN_TXT_PATH = './data/simpletrain.txt'
TEXT_DES_PATH = './data/txt'
IMAGE_DES_PATH = './data/img'

if not os.path.exists(TEXT_DES_PATH):
    os.mkdir(TEXT_DES_PATH)
if not os.path.exists(IMAGE_DES_PATH):
    os.mkdir(IMAGE_DES_PATH)


def copy_file(src, txt_des, img_des):


    files = os.listdir(FILES_PATH)
    txts = []

    imgs = []

    for i in files:
        ext = i.split('.')[-1]
        if ext == 'txt':
            txts.append(i)
        else:
            imgs.append(i)

    for t in txts:
        shutil.copy(src + '/' + t, txt_des + '/' + t)

    for i in imgs:
        shutil.copy(src + '/' + i, img_des + '/' + i)

    print('File copy is done.')

def create_simple_train_file(filepath, txt_src, img_src):
    txts = os.listdir(txt_src)
    imgs = os.listdir(img_src)

    for t in txts:
        img_path = ''

        for i in imgs:
            if i[:-4] == t[:-4]:
                img_path = i
                break

        if img_path is '':
            continue

        with open(txt_src + '/' + t, 'r') as f:
            while True:
                line = f.readline()
                if line is None or line == '':
                    break
                print(line)
                data = line.split(' ')

                classes = data[0]
                x1 = int(float(data[1]) * 512)
                y1 = int(float(data[2]) * 512)
                x2 = int(float(data[3]) * 512)
                y2 = int(float(data[4]) * 512)

                class_name = ''

                if classes == '0':
                    class_name = 'crack'
                elif classes == '1':
                    class_name = 'horizon crack'
                elif classes == '2':
                    class_name = 'vertical crack'
                else:
                    class_name = 'step crack'

                with open(filepath, 'a') as fe:
                    one_line = img_src + '/' + img_path + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + class_name + '\n'
                    fe.write(one_line)

import cv2 as cv

def change_imgsize(filepath):
	imgs = os.listdir(filepath)

	cutoff = 0
	count = 0

	for i in imgs:
		image = cv.imread(filepath + '/' + i)

		if count == 0:
			image = cv.resize(image, (92, 92))
		elif count == 1 or count == 2:
			image = cv.resize(image, (152, 152))
		else:
			image = cv.resize(image, (196, 196))


		cv.imwrite(filepath + '/' + i, image)

		cutoff += 1

		if cutoff >= 5000:
			count += 1
			cutofff = 0

def create_train_data(filepath):
    files = os.listdir(filepath)

    for f in files:
        image = cv.imread(filepath + '/' + f)

        h, w = image.shape[:2]

        image2 = cv.resize(image, None, fx=0.5, fy=1, interpolation=cv.INTER_AREA)
        image3 = cv.resize(image, None, fx=1, fy=0.5, interpolation=cv.INTER_AREA)
        image4 = cv.resize(image, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
        cv.imwrite(filepath + '/x05y1_image_' + f, image2)
        cv.imwrite(filepath + '/x1y05_image_' + f, image3)
        cv.imwrite(filepath + '/x05y05_image_' + f, image4)

        filp_image = cv.flip(image, 1)
        cv.imwrite(filepath + '/f_image_' + f, filp_image)
        image2 = cv.resize(filp_image, None, fx=0.5, fy=1, interpolation=cv.INTER_AREA)
        image3 = cv.resize(filp_image, None, fx=1, fy=0.5, interpolation=cv.INTER_AREA)
        image4 = cv.resize(filp_image, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
        cv.imwrite(filepath + '/fx05y1_image_' + f, image2)
        cv.imwrite(filepath + '/fx1y05_image_' + f, image3)
        cv.imwrite(filepath + '/fx05y05_image_' + f, image4)


def create_edge_image(filepath, desPath):

    files = os.listdir(filepath)

    totalTime = 0

    if not os.path.exists(desPath):
        os.mkdir(desPath)

    for f in files:
        start = time.time()
        image = cv.imread(filepath + '/' + f)
        copy = image.copy()
        image2 = cv.cvtColor(copy, cv.COLOR_BGR2GRAY)
        # Image read and convert Gray Level.

        blur = cv.GaussianBlur(image2, (3, 3), 0)
        # It is need to run canny edge function
        canny = cv.Canny(blur, 60, 255)
        # canny edge detection

        cv.imwrite(desPath + '/' + f, canny)
        end = time.time()

        totalTime += (end - start)

    print('Done : average time : ', totalTime / 20000)


def divide_data_set(filepath, type):
    files = os.listdir(filepath)
    limit = 5000
    count = 1
    saveIndex = 0

    images = []

    if type == 'binary':
        for f in files:

            if count >= limit:
                np.save('crack_train_ne' + str(saveIndex) + '.npy', images)
                images.clear()
                saveIndex += 1
                count = 1

            image = cv.imread(filepath + '/' + f, cv.IMREAD_GRAYSCALE)
            images.append(image)
            count += 1

def write_train_file(imgpath, txtpath, type):
	imgfiles = os.listdir(imgpath)

	with open(txtpath , 'a') as fe:
		for f in imgfiles:
			img = cv.imread(imgpath + '/' + f)

			x1 = 0
			y1 = 0

			x2, y2, _ = img.shape

			x1 = str(x1 + 1)
			y1 = str(y1 + 1)
			x2 = str(x2 - 1)
			y2 = str(y2 - 1)

			one_line = imgpath + '/' + f + ',' + x1 + ',' + y1 + ',' + x2 + ',' + y2 + ',' + type +'\n'

			fe.write(one_line)


def make_csv_file(path, class_label):
    datas = os.listdir(path)

    with open('./classification_file_kaggle.csv', 'a') as f:
        for d in datas:
            line = path + '/' + d + ',' + str(class_label) + '\n'

            f.write(line)

#make_csv_file('./cracks/Positive', 0)
#make_csv_file('./cracks/Negative', 1)
# make_csv_file('./SDNET2018/D/CD', 0)
# make_csv_file('./SDNET2018/D/UD', 1)
# make_csv_file('./SDNET2018/P/CP', 2)
# make_csv_file('./SDNET2018/P/UP', 3)
# make_csv_file('./SDNET2018/W/CW', 4)
# make_csv_file('./SDNET2018/W/UW', 5)
#create_train_data('./cracks/fixed_sky')
#change_imgsize('./cracks/Negative')
#change_imgsize('./cracks/Positive')
#write_train_file('./cracks/Positive', './trainData/simpletrain.txt', 'coarse crack')
# write_train_file('./SDNET2018small/P/CP', './trainData/simpletrain.txt', 'coarse crack')
# write_train_file('./SDNET2018small/D/CD', './trainData/simpletrain.txt', 'find crack')
# write_train_file('./SDNET2018small/W/CW', './trainData/simpletrain.txt', 'inclusion crack')
# write_train_file('./cracks/Positive192', './trainData/simpletrain.txt', 'crack')
write_train_file('./SDNET2018/P/CP', './trainData/simpletrain.txt', 'coarse crack')
write_train_file('./SDNET2018/D/CD', './trainData/simpletrain.txt', 'find crack')
write_train_file('./SDNET2018/W/CW', './trainData/simpletrain.txt', 'inclusion crack')
# change_imgsize('./SDNET2018small/P/CP')
# change_imgsize('./SDNET2018small/P/UP')
# change_imgsize('./SDNET2018small/D/CD')
# change_imgsize('./SDNET2018small/D/UD')
# change_imgsize('./SDNET2018small/W/CW')
# change_imgsize('./SDNET2018small/W/UW')
#create_temp('./cracks/fixed_sky', './trainData/simpletrain.txt')
#create_temp('./cracks/Positive', './trainData/simpletrain.txt')
#create_simple_train_file(TRAIN_TXT_PATH, TEXT_DES_PATH, IMAGE_DES_PATH)
#create_edge_image('./cracks/Positive', './cracks/edge_positive')
#create_edge_image('./cracks/Negative', './cracks/edge_negative')
#divide_data_set('./cracks/edge_positive', type='binary')
#divide_data_set('./cracks/edge_negative', type='binary')