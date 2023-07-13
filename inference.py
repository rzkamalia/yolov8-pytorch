import cv2

from config import *

if MODE == 0:
    from detect import BasePredictor, drawPropertiesResult
elif MODE == 1:
    from segment import BasePredictor

video = cv2.VideoCapture('pedes.mp4')
while True:
    ret, frame = video.read()
    if not ret:
        video = cv2.VideoCapture('pedes.mp4')
    else:
        base_predictor = BasePredictor()
        result_detector = base_predictor.stream_inference(frame)
        for result in result_detector:
            drawPropertiesResult(result, frame)

        cv2.imshow('RESULT', frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()        