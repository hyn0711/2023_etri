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

# socket에서 수신한 버퍼를 반환하는 함수 #
def recvall(sock, count):
# 바이트 문자
    buf = b''
    while count:
        newbuf = sock.recv(count)    #소켓으로부터 데이터를 읽음, 최대 count 바이트만큼의 데이터를 읽어옴, 읽어드릴 데이터가 없으면 상대방이 데이터를 보내줄 때 까지 대기
        if not newbuf: return None  
        buf += newbuf                # buf = buf + newbuf
        count -= len(newbuf)         # count = count - len(newbuf) , len() : 문자열의 길이 반환
    return buf
 
HOST='129.254.187.105'
PORT=8485
 
server_socket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')
 
server_socket.bind((HOST,PORT))
print('Socket bind complete')

server_socket.listen(10)
print('Socket now listening')
 
client_socket,addr=server_socket.accept()
 
while True:

    # client에서 받은 stringData의 크기 (==(str(len(stringData))).encode().ljust(16))
    #ljust : 문자열을 왼쪽으로 16만큼 정렬
    length = recvall(client_socket, 16)       #client_socket으로부터 수신한 버퍼를 최대 16바이트씩 반환
    stringData = recvall(client_socket, int(length))   
    data = np.fromstring(stringData, dtype = 'uint8')  
    
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)  #data를 디코딩

    image = np.array(frame)   #디코딩된 frame을 다시 numpy.array하여 행렬 형태로..?
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
    
    client_socket.sendall(output)
    cv2.waitKey(1)
server_socket.close()
client_socket.close()
    
