import os
import shutil

FILES_PATH = 'C:/Users/boss/Desktop/YOLO_M~1/x64/Release/data/complete'
TRAIN_TXT_PATH = './trainData/simpletrain.txt'
TEXT_DES_PATH = './trainData/txt'
IMAGE_DES_PATH = './trainData/img'

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



#copy_file(FILES_PATH, TEXT_DES_PATH, IMAGE_DES_PATH)
create_simple_train_file(TRAIN_TXT_PATH, TEXT_DES_PATH, IMAGE_DES_PATH)