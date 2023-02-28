#! /usr/bin/env python3

import time
import subprocess
import sys

from vosk import Model, KaldiRecognizer, SetLogLevel

from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
from luma.core.virtual import viewport
from luma.core.legacy import text, show_message
from luma.core.legacy.font import proportional, CP437_FONT, TINY_FONT, SINCLAIR_FONT, LCD_FONT


serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, cascaded=4, block_orientation=90, blocks_arranged_in_reverse_order=True)

while True:
    record = ['arecord','-D','plughw:3,0','--duration','5','out.wav']

    p = subprocess.Popen(record, shell=False)
    time.sleep(5)
    
    SAMPLE_RATE = 16000

    SetLogLevel(0)

    model = Model("model")
    rec = KaldiRecognizer(model, SAMPLE_RATE)

    with subprocess.Popen(["ffmpeg", "-loglevel", "quiet", "-i",
                                'out.wav',
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
        FinalResult = FinalResult.replace("{","")
        Final_Result = FinalResult.replace("}","")
        print(Final_Result)

    show_message(device, Final_Result, fill="white", font=proportional(LCD_FONT), scroll_delay=0.1)
            
        
        
