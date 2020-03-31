import cv2
import socket
import numpy as np
from queue import Queue
from _thread import*

enclosure_queue = Queue()

def webcam(queue):
    capture = cv2.VideoCapture(0)
    capture.set(3, 320);
    capture.set(4, 240);

	while True:
		ret, frame = capture.read()
		if ret == False:
			continue
		encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]

        imgencode = cv2.imencode('.jpg', frame, encode_param)

        data = np.array(imgencode)
		stringData = data.tostring()
		queue.put(stringData)

        stringDataget = queue.get()
		client_socket.sendall(str(len(stringDataget)).ljust(16).encode())
		client_socket.sendall(stringDataget)

		cv2.imshow('Rpi-streaming', frame)


		key = cv2.waitKey(1)

		if key == 27:
			break


HOST = '127.0.0.1'
PORT = 9000

client_socket= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))
webcam(enclosure_queue)
client_socket.close()
