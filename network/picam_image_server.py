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
 
HOST='129.254.187.105'
PORT=8485
 
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')
 
s.bind((HOST,PORT))
print('Socket bind complete')

s.listen(10)
print('Socket now listening')
 
conn,addr=s.accept()
 
while True:

    length = recvall(conn, 16)
    stringData = recvall(conn, int(length))
    data = np.fromstring(stringData, dtype = 'uint8')
    
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    cv2.imshow('ImageWindow',frame)
    cv2.waitKey(1)
