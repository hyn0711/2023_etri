# image_socket_server

import socket
import struct
import pickle
import cv2

ip = '129.254.187.105'
port = 5050

server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

server_socket.bind((ip, port))

server_socket.listen(10)
print('클라이언트 연결 대기')

client_socket, address = server_socket.accept()
print('클라이언트 ip 주소 :', address[0])

data_buffer = b""

data_size = struct.calcsize("L")

while True:
    while len(data_buffer) < data_size:
        data_buffer += client_socket.recv(4096)

    packed_data_size = data_buffer[:data_size]
    data_buffer = data_buffer[data_size:]

    frame_size = struct.unpack(">L", packed_data_size)[0]

    while len(data_buffer) < frame_size:
        data_buffer += client_socket.recv(4096)

    frame_data = data_buffer[:frame_size]
    data_buffer = data_buffer[frame_size:]

    print("수신 프레임 크기 : {} bytes".format(frame_size))

    frame = pickle.loads(frame_data)

    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q")Z:
        break

client_socket.close()
server_socket.close()
print('연결 종료')

cv2.destroyAllWindows()
