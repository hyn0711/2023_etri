# image_socket_client
import cv2 
import socket
import pickle 
import struct 

ip = '129.254.187.105'
port = 5050


capture = cv2.VideoCapture(0)


capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
    
    client_socket.connect((ip, port))
    
    print("연결 성공")
    
    while True:
        
        retval, frame = capture.read()
        retval, frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        frame = pickle.dumps(frame)

        print("전송 프레임 크기 : {} bytes".format(len(frame)))
               client_socket.sendall(struct.pack(">L", len(frame)) + frame)

capture.release()
