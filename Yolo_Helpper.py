import os
import numpy as np
import random
import shutil

def yolo3_image_info_create(filePath, classLabel, desPath):
    '''
    yolo3 학습을 위한, txt 파일 생성 메소드
    :param filePath: 폴더 경로
    :param classLabel: 적용할 클래스 라벨 0, 1, 2, 3, 4, 5
    :param desPath: txt가 저장될 경로
    '''
    images = os.listdir(filePath)

    one_class = classLabel
    one_x1 = 0.01
    one_x2 = 0.97
    one_y1 = 0.01
    one_y2 = 0.97

    print('Start writing...')
    for i in images:
        with open(desPath + '/' + i[:-4] + '.txt', 'w') as f:
            line = "{} {} {} {} {}\n".format(one_class, one_x1, one_y1, one_x2, one_y2)
            f.write(line)

        shutil.copy(filePath + '/' + i, desPath + '/' + i)
    print('Done...')

def delete_no_data_txt(filepath):
    files = os.listdir(filepath)
    txts = []

    print("start delete task!")
    line = ''

    for f in files:
        if f[-4:] != '.txt':
            continue

        with open(filepath + '/' + f) as t:
            line = t.readline()

        if not line:
            os.remove(filepath + '/' + f)
            os.remove(filepath + '/' + f[:-4] + '.jpg')

    print('Done...')


def dataset_divide(filepath, train=0.90, valid=0.2):

    files = os.listdir(filepath)
    txts = []

    print('Divide Start...')

    for f in files:
        if f[-4:] == '.txt':
            txts.append('data/obj/' + f[:-4] + '.jpg')

    del files

    random.seed(100)

    split_size = int(len(txts) * train)
    print('Total txt : ', len(txts))
    print('Split_size : ', split_size)
    permuted = np.random.permutation(txts)
    train, test = permuted[:split_size], permuted[split_size:]

    print('Train : ', len(train))
    print('Test : ', len(test))

    with open('D:/Anaconda3/Scripts/CrackDetection/darknet-master/build/darknet/x64/data/train.txt', 'w') as f:
        for tr in train:
            line = tr + '\n'
            f.write(line)

    with open('D:/Anaconda3/Scripts/CrackDetection/darknet-master/build/darknet/x64/data/test.txt', 'w') as f:
        for te in test:
            line = te + '\n'
            f.write(line)

    print('Done...')

#yolo3_image_info_create('D:\Anaconda3\Scripts\CrackDetection\cracks\Positive', 2, 'D:/Anaconda3/Scripts/CrackDetection/darknet-master/build/darknet/x64/data/obj')
#delete_no_data_txt('D:/Anaconda3/Scripts/CrackDetection/yolo_mark/x64/Release/data/img')
# yolo3_image_info_create('C:/Users/hwchoi/Desktop/SDNET2018/P/CP', 1, 'D:/Anaconda3/Scripts/CrackDetection/darknet-master/build/darknet/x64/data/obj')
# yolo3_image_info_create('C:/Users/hwchoi/Desktop/SDNET2018/W/CW', 2, 'D:/Anaconda3/Scripts/CrackDetection/darknet-master/build/darknet/x64/data/obj')
dataset_divide('D:/Anaconda3/Scripts/CrackDetection/darknet-master/build/darknet/x64/data/obj')