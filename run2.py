#! /usr/bin/env python3

import time
import subprocess
import sys
import threading
from multiprocessing import Process

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

from run1 import speech, output, frame1
from vosk import Model, KaldiRecognizer, SetLogLevel

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


client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('129.254.187.105', 8485))



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
        
    
def yolov7():
    while True:
        
        if hello == "hello":
            # reshape output data (change dim)
            if not output.shape[0] == 0:
                newdim = int(output.shape[0]/58)
                output = output.reshape(newdim, 58)
    
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
    
            if output.shape[0] > 0:
                for idx in range(output.shape[0]):
                    plot_skeleton_kpts(frame1, output[idx, 7:].T, 3)
    
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        
        else:
            break
        
    
        
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]



def Speech_to_text():

    while True:
        '''
        record = ['arecord','-D','plughw:2,0','--duration','5','out.wav']

        p = subprocess.Popen(record, shell=False)
        time.sleep(5)
        '''
        
    
        SAMPLE_RATE = 16000

        SetLogLevel(0)

        model = Model("model")
        rec = KaldiRecognizer(model, SAMPLE_RATE)

        with subprocess.Popen(["ffmpeg", "-loglevel", "quiet", "-i",
                                    'hello.wav',
                                    "-ar", str(SAMPLE_RATE) , "-ac", "1", "-f", "s16le", "-"],
                                    stdout=subprocess.PIPE) as process:

            while True:
                data = process.stdout.read(4000)
                
                if len(data) == 0:
                    break
                    
                if rec.AcceptWaveform(data):
                    print(rec.Result())
                    
                else:
                    print(rec.PartialResult())
                   
            #print(rec.FinalResult())
            Final_Result = rec.FinalResult()
            #FinalResult = FinalResult.replace(" ","")
            #FinalResult = FinalResult.replace("{","")
            #FinalResult = FinalResult.replace("}","")
            #FinalResult = FinalResult.replace("text","")
            #Final_Result = FinalResult.replace(":","")
            #Final_Result = FinalResult.replace('"','')
            #print(Final_Result)
            
            return Final_Result
        
        
while True:
    start = time.time()
    
    
    hello = "hello"
    
    print(speech)
    if hello in speech:
        
        
        
        # reshape output data (change dim)
        if not output.shape[0] == 0:
            newdim = int(output.shape[0]/58)
            output = output.reshape(newdim, 58)
            
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
        
        if output.shape[0] > 0:
            for idx in range(output.shape[0]):
                plot_skeleton_kpts(frame1, output[idx, 7:].T, 3)
        
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        end = time.time()
        fps = 1 /(end-start)
        start = end
    
        cv2.putText(frame1, "%d detected"%output.shape[0], (7,70), cv2.FONT_ITALIC, 1, (51,51,51), 2, cv2.LINE_AA)
        
    
        cv2.imshow('pose', frame1)
        cv2.waitKey(1)
        
    else:
        print("no")

client_socket.close()
