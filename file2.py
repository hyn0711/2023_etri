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

from file1 import output, frame1
from vosk import Model, KaldiRecognizer, SetLogLevel

from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
from luma.core.virtual import viewport
from luma.core.legacy import text, show_message
from luma.core.legacy.font import proportional, CP437_FONT, TINY_FONT, SINCLAIR_FONT, LCD_FONT

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
        FinalResult = rec.FinalResult()
        FinalResult = FinalResult.replace(" ","")
        FinalResult = FinalResult.replace("{","")
        FinalResult = FinalResult.replace("}","")
        FinalResult = FinalResult.replace("text","")
        FinalResult = FinalResult.replace(":","")
        Final_Result = FinalResult.replace('"','')
        #print(Final_Result)
            
        return Final_Result
      	#print(Final_Result)
		
	if Final_Result == "hello":
		if not output.shape[0] == 0:
			newdim = int(output.shape[0]/58)
			output = output.reshape(newdim, 58)
			
		frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
			
		if output.shape[0] > 0:
			for idx in range(output.shape[0]):
				plot_skeleton_kpts(frame1, output1[idx, 7:].T, 3)
					
		frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
			
	else:
		print("no yolo")
			
			
	
