import cv2
import socket
import numpy as np
from picamera2 import Picamera2

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

client_socket.connect(('129.254.187.105', 8485))
 
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
 
while True:

    frame1 = picam2.capture_array()
    frame = frame1.copy()
    result, frame = cv2.imencode('.jpg', frame, encode_param)

    data = np.array(frame)
    stringData = data.tostring()

    client_socket.sendall((str(len(stringData))).encode().ljust(16) + stringData)
    
    output1 = client_socket.recv(4096)
    output = np.fromstring(output1, dtype = 'float64')
    
    print(output)
    
    #output = client_socket.recv(16)
    #print(output)
    
client_socket.close()
