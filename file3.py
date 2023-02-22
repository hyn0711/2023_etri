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

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('129.254.187.105', 8485))


# picamera 
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640,360)}))
picam2.start()

while True:
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
				
		FinalResult = rec.FinalResult()
		FinalResult = FinalResult.replace(" ","")
		FinalResult = FinalResult.replace("{","")
		FinalResult = FinalResult.replace("}","")
		FinalResult = FinalResult.replace("text","")
		FinalResult = FinalResult.replace(":","")
		Final_Result = FinalResult.replace('"','')
        
       
	if Final_Result == "hello":
		if not output.shape[0] == 0:
			newdim = int(output.shape[0]/58)
			output = output.reshape(newdim, 58)
			
		frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
		
		if output.shape[0] > 0:
			for idx in range(output.shape[0]):
				plot_skeleton_kpts(frame1, output[idx, 7:].T, 3)
				
		frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
		
	else:
		print("no yolo")
		
	cv2.putText(frame1, "%d detected"%output.shape[0], (7,70), cv2.FONT_ITALIC, 1, (51,51,51), 2, cv2.LINE_AA)
