import socket
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import utils
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weigths = torch.load('yolov7-w6-pose.pt', map_location=device)
model = weigths['model']
_ = model.float().eval()

if torch.cuda.is_available():
    print("cuda : available")
    model.half().to(device)

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
 
server_socket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')
 
server_socket.bind((HOST,PORT))
print('Socket bind complete')

server_socket.listen(10)
print('Socket now listening')
 
client_server,addr=server_socket.accept()
 
while True:

    length = recvall(client_server, 16)
    stringData = recvall(client_server, int(length))
    data = np.fromstring(stringData, dtype = 'uint8')
    
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

    image = np.array(frame)
    start = time.time()
    image = letterbox(image, 1280, stride=64, auto=True)[0]
    with torch.no_grad():
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))
        if torch.cuda.is_available():
            image = image.half().to(device)
        output, _ = model(image)
        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
        output = output_to_keypoint(output)
        
        
    #cv2.imshow('ImageWindow',frame)
    pirnt(output)
    #print(type(output))
    
    output1 = output.tostring()
    client_socket.sendall(output1)
    
    #client_socket.sendall(output1)
    cv2.waitKey(1)

    server_socket.close()
    client_socket.close()
