import cv2

from config import *
from predictor import BasePredictor


video = cv2.VideoCapture(source_input)
while True:
    ret, frame = video.read()
    if not ret:
        video = cv2.VideoCapture(source_input)
    else:
        base_predictor = BasePredictor()
        result_detector = base_predictor.stream_inference(frame)

        cv2.imshow('RESULT', result_detector)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()        