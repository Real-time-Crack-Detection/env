import cv2
import socket
import numpy as np
import time

def receiving():
   s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

   HOST = '121.133.178.126'
   PORT = 30000
   s.connect((HOST,PORT))

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
   count = 0
   while True:

      start = time.time()
      length = recvall(s, 16)
      #client recv part
      stringData = recvall(s, int(length))

#      print('Image length receive time : ', time.time() - start)

      data = np.frombuffer(stringData, dtype='uint8')

      result = cv2.imdecode(data, cv2.IMREAD_COLOR)

      cv2.imshow('result', result)
      count += 1
      if count % 2 == 0:
         cv2.imwrite("static/img/realtime.jpg", result)
      elif count % 2 == 1:
         cv2.imwrite("static/img/realtime2.jpg", result)

      if count > 100:
         count = 0
      cv2.waitKey(1)

      print(length)


   cam.release()