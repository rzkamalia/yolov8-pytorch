# Ultralytics YOLO 🚀, AGPL-3.0 license

import cv2
import random
import numpy as np
import torch

from autobackend import AutoBackend
from utils import LetterBox, non_max_suppression, scale_boxes, xyxy2xywh

from config import *

if torch.cuda.is_available():
    device_name = 'cuda'
    print('Application is utilizing GPU (CUDA)...')
else:
    device_name = 'cpu'
    print('Application is utilizing CPU...')

# Setup model. Initialize YOLO model with given parameters and set it to evaluation mode.
device = torch.device(device_name)
model = AutoBackend(device=device, fp16=False, fuse=True)
model.eval()
    
class BasePredictor:

    def preprocess(self, im, model, device):
        """Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """

        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        img = im.to(device)
        img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
        if not_tensor:
            img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def pre_transform(self, im):
        """Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Return: A list of transformed imgs.
        """

        return [LetterBox(auto=True)(image=im)]

    def postprocess(self, preds, img, orig_imgs):
        """Postprocesses predictions and returns a list of Results objects."""

        preds = non_max_suppression(preds[0])

        results = []
        for pred in preds:
            pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_imgs.shape)
            for *xyxy, conf, cls in reversed(pred):
                xywh = xyxy2xywh(torch.tensor(xyxy).view(1, 4))
                label = cls.cpu().detach()
                score = conf.cpu().detach()
                results.append(dict(boxes = xywh, labels = label, scores = score))
        return results

    def stream_inference(self, frame=None):
        """Streams real-time inference on camera feed and saves results to file."""

        im = self.preprocess(frame, model, device)
        with torch.no_grad():
            preds = model(im, augment=False, visualize=False)
        results = self.postprocess(preds, im, frame)
        return results
    

def getCoordinateForDrawBox(box):
    x1 = int(box[0])
    y1 = int(box[1])
    wx = int(box[2])
    hy = int(box[3])

    x2 = int(x1 + wx)
    y2 = int(y1 + hy)
    return (x1, y1, x2, y2)


def drawPropertiesResult(result, frame):
    box = result['boxes']
    label = class_dict.get(int(result['labels']))
    conf = result['scores']

    random.seed(int(result['labels']))
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    for box in box:
        (x1, y1, x2, y2) = getCoordinateForDrawBox(box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, str(label) + ": " + str(round(conf.item(), 4)), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)