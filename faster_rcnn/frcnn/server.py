import socket
import cv2
import numpy as np

def recvall(sock, count):
	buf = b''
	while count:
		newbuf = sock.recv(count)
		if not newbuf: return None
		buf += newbuf
		count -= len(newbuf)
	return buf

HOST = '192.168.0.13'
PORT = 30000

#TCP

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('socket created')

server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

server_socket.bind((HOST, PORT))
print('Socket bind complete')


server_socket.listen()
print('server listening....')
client_socket, addr = server_socket.accept()
print('Connected by', addr)


fcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter("test-1.avi", fcc, 30.0, (320, 240))
while True:
	length = recvall(client_socket,16)
	stringData = recvall(client_socket,int(length))
	data = np.frombuffer(stringData, dtype = 'uint8')
	decimg=cv2.imdecode(data,1)
	cv2.imshow('server_streaming',decimg)

	writer.write(decimg)
	key = cv2.waitKey(1)
	if key == 27:
          break

client_socket.close()
server_socket.close()