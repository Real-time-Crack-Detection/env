from keras.models import load_model
import numpy as np
import Malware_Classification.Malware_Classification_Method as Method
import os
import cv2


#MODEL_PATH = './model/Judge_Model.hdf5'
MODEL_PATH = './model/Judge_Model_v_1.1.hdf5'
BYTES_SAVE_PATH = './bytes/'
IMAGE_SIZE = 256

class Malware_Classification_App:

    def __init__(self):
        self.App_start()

    def close(self):
        self.exit = True

    def App_start(self):
        #tf.ConfigProto().gpu_options.allow_growth=True
        self.exit = False
        # thread에 요청이 왔을 때만, 수행하기 위한 변수

        self.model = load_model(MODEL_PATH)
        # keras 모델 정보를 읽어들임

        if not os.path.exists(BYTES_SAVE_PATH):
            os.mkdir(BYTES_SAVE_PATH)

    def judge(self, path):
        # 실제로 판단하는 모듈
        hex_data = Method.read_file(path)
        # File의 bytes값을 읽어들임.

        filename = path.split('/')[-1]
        bytes_file = Method.create_bytes_file(BYTES_SAVE_PATH, filename, hex_data)
        # bytes값을 .txt 형태로 저장

        obj_img = Method.bytes2png(bytes_file, IMAGE_SIZE, '.')

        # .txt의 파일을 Image화 시키는 과정
        obj_img = cv2.resize(cv2.imread(obj_img, cv2.IMREAD_GRAYSCALE),
                           (IMAGE_SIZE, IMAGE_SIZE))
        obj_img = obj_img / 255
        obj_img = np.column_stack([obj_img.flatten()])
        obj_img = np.reshape(obj_img, [IMAGE_SIZE, IMAGE_SIZE, 1])
        # 실제 데이터를 np.array에서 Image로 바꾸는 과정
        # shape (256, 256, 1)
        obj_img = np.expand_dims(obj_img, axis=0)
        # 실제 모델에 들어가기 위해선 tensor(4차원)의 형태여야함.
        # shape(1, 256, 256, 1)

        decoded_img = self.model.predict(obj_img)
        # 모델에 해당 Image 파일 예측 -> decode된 np.array 값 반환

        error_value = np.sum(obj_img) - np.sum(decoded_img)
        # 실제 이미지와 예측 이미지의 오차값을 픽셀별로 계산함.
        error_value = abs(error_value)
        print('error : ', error_value)


        if error_value > 500.0:
            return 0
        # 아무 이상 없는 경우
        elif error_value > 200.0 and error_value <= 500.0:
            return 1
        # 주의해야 할 필요가 있는 경우
        else:
            return 2
        # 악성코드로 의심되는 경우