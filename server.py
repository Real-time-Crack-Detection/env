import socket
import cv2
import numpy as np
import time
from crack_manager import CrackManager
import tensorflow as tf
# import required packages
import cv2
import numpy as np


WEIGH_PATH = './yolo_model/tiny-yolo-crack-deep_best.weights'
CONFIG_PATH = './yolo_model/tiny-yolo-crack-deep.cfg'
CLASS_PATH = './yolo_model/obj.names'



def predict_image(imagePath, scale):
    import time
    # read input image
    # image = cv2.imread(imagePath)
    image = imagePath

    Width = image.shape[1]
    Height = image.shape[0]

    start = time.time()

    blob = cv2.dnn.blobFromImage(image, scale, (Width, Height), (0, 0, 0), True, crop=False)
    net.setInput(blob)


    # function to get the output layer names
    # in the architecture
    def get_output_layers(net):
        layer_names = net.getLayerNames()

        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        return output_layers


    # function to draw bounding box on the detected object with class name
    def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(classes[class_id])

        color = COLORS[class_id]

        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    # run inference through the network
    # and gather predictions from output layers
    outs = net.forward(get_output_layers(net))

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.7
    nms_threshold = 0.7

    # for each detetion from each output layer
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # go through the detections remaining
    # after nms and draw bounding box
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))


    print('image predicting time : ', time.time() - start)
    # display output image

    return image
    # save output image to disk
    #cv2.imwrite("object-detection.jpg", image)

    # release resources
    #cv2.destroyAllWindows()



# read class names from text file
classes = None
with open(CLASS_PATH, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# generate different colors for different classes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# read pre-trained model and config file
net = cv2.dnn.readNet(WEIGH_PATH, CONFIG_PATH)



configue = tf.ConfigProto()
configue.gpu_options.allow_growth = True
sess = tf.Session(config=configue)

#c = CrackManager('./model_frcnn_hdf5', './config.pickle', './classification_model_v1_0_kaggle.hdf5')

# socket에서 수신한 버퍼를 반환하는 함수
def recvall(sock, count):
   # 바이트 문자열
   buf = b''
   while count:
      newbuf = sock.recv(count)
      if not newbuf: return None
      buf += newbuf
      count -= len(newbuf)
   return buf


HOST = '192.168.0.8'
#HOST = '172.30.34.18'
PORT = 30000

# TCP 사용
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

# 서버의 아이피와 포트번호 지정
s.bind((HOST, PORT))
print('Socket bind complete')
# 클라이언트의 접속을 기다린다. (클라이언트 연결을 n개까지 받는다)
s.listen(1)
print('Socket now listening')

# 연결, conn에는 소켓 객체, addr은 소켓에 바인드 된 주소
conn, addr = s.accept()
print('ras accept')

conn2, addr2 = s.accept()
print('web accept')
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]





while True:
   start = time.time()
   # client에서 받은 stringData의 크기 (==(str(len(stringData))).encode().ljust(16))
   length = recvall(conn, 16)
   stringData = recvall(conn, int(length))

   print('Image length receive time : ', time.time() - start)

   start = time.time()
   #data = np.fromstring(stringData, dtype='uint8')
   data = np.frombuffer(stringData, dtype='uint8')
   # data를 디코딩한다.

   frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
   print('Image length decoding time : ', time.time() - start)

   print('Image shape : ', frame.shape)

   try:
      #img = predict_image(frame, 0.015)
      # 현재 찾은 최적의 스케일 값 0.009 ~ 0.015
      img = predict_image(frame, 0.0095)

      cv2.imshow('ImageWindow', img)
      cv2.waitKey(1)



      # 전송 부분
      result, result_img = cv2.imencode('.jpg', frame, encode_param)

      data = np.array(result_img)
      stringData = data.tostring()

      # server send
      length = (str(len(stringData))).encode('utf-8').ljust(16)

      conn2.sendall(length + stringData)
   except Exception as e:
      print(e)

##### 해당 부분까지는 이미지를 읽어서 추론하는 부분 #####

# VideoCapture 객체를 메모리 해제하고 모든 윈도우 창을 종료합니다.
cam.release()
writer.release()
cv2.destroyAllWindows()