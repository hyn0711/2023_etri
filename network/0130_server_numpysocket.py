import logging

from numpysocket import NumpySocket
import cv2

logger = logging.getLogger('OpenCV server')
logger.setLevel(logging.INFO)

with NumpySocket() as s:
    s.bind(('129.254.187.105', 9999))
    
    while True:
      try:
        s.lesten()
        conn, addr = s.accept()
        
        logger.info(f"connected: {addr}")
        
        while conn:
          frame = conn.recv()
          if len(frame) == 0:
            break
            
            cv2.imshow('Frame', frame)
            
        logger.info(f"disconnected: {addr}")
    except ConnectionResetError:
      pass
          
          
