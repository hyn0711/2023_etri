import cv2
import socket
import numpy as np
from picamera2 import Picamera2
import warnings
import time
import torch
import pickle
import re
import argparse
import sys
import asyncio

from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
from luma.core.virtual import viewport
from luma.core.legacy import text, show_message
from luma.core.legacy.font import proportional, CP437_FONT, TINY_FONT, SINCLAIR_FONT, LCD_FONT

def recvall(sock, count):
     buf = b''
     while count:
         newbuf = sock.recv(count)
         if not newbuf: return None
         buf += newbuf
         count -= len(newbuf)
     return buf

# HOST,PORT file input
f = open("host_port.txt", "r", encoding = "utf-8")
line = f.readlines()
HOST = line[0].strip()
HOST = HOST[7:]
PORT = line[1]
PORT = PORT[7:]
PORT = int(PORT)

warnings.filterwarnings('ignore')

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))
#client_socket.connect(('129.254.187.105', 8485))

# picamera 
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640,360)}))
picam2.start()

# create LED matrix device
serial = spi(port=1, device=0, gpio=noop())
device = max7219(serial, cascaded=4, block_orientation=90, blocks_arranged_in_reverse_order=True)


    
def plot_skeleton_kpts(im, kpts, steps, orig_shape=None):
    #Plot the skeleton and keypointsfor coco datatset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    num_kpts = len(kpts) // steps

    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < 0.5:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)

    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        if steps == 3:
            conf1 = kpts[(sk[0]-1)*steps+2]
            conf2 = kpts[(sk[1]-1)*steps+2]
            if conf1<0.5 or conf2<0.5:
                continue
        if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)


encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

while True:
    start = time.time()
    
    # send frame (client -> server)
    frame1 = picam2.capture_array()
    frame = frame1.copy()
    result, frame = cv2.imencode('.jpg', frame, encode_param)
    data = np.array(frame)
    stringData = data.tostring()
    client_socket.sendall((str(len(stringData))).encode().ljust(16) + stringData)
    
    # receive output (server -> client)   
    length = recvall(client_socket,16)
    stringData_ = recvall(client_socket, int(length))    
    output = np.fromstring(stringData_, dtype='float64')
    
    # reshape output data (change dim)
    if not output.shape[0] == 0:
        newdim = int(output.shape[0]/58)
        output = output.reshape(newdim, 58)
    
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
    
    if output.shape[0] > 0:
        for idx in range(output.shape[0]):
            plot_skeleton_kpts(frame1, output[idx, 7:].T, 3)
    
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    
    # LED text code
    
    with canvas(device) as draw:
        text(draw, (16, 1), "%d"%int(output.shape[0]//10), fill="white", font=proportional(CP437_FONT))
        text(draw, (24, 1), "%d"%int(output.shape[0]%10), fill="white", font=proportional(CP437_FONT))
    
    #show_message(device, "%d detected"%output.shape[0], fill="white", font=proportional(LCD_FONT), scroll_delay=0.1)
    
    end = time.time()
    fps = 1 /(end-start)
    start = end
    
    cv2.putText(frame1, "%d detected"%output.shape[0], (7,70), cv2.FONT_ITALIC, 1, (51,51,51), 2, cv2.LINE_AA)
    cv2.putText(frame1, "FPS : %0.3f"%fps, (7,40), cv2.FONT_ITALIC, 1, (51,51,51), 2, cv2.LINE_AA )
    
    cv2.imshow('pose', frame1)
    cv2.waitKey(1)
client_socket.close()

