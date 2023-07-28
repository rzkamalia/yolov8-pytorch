# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np
import torch

from config import MODE
from autobackend import AutoBackend
from result import Results
from utils import LetterBox, non_max_suppression, scale_boxes, process_mask


if torch.cuda.is_available():
    device_name = 'cuda'
    print('cuda')
else:
    device_name = 'cpu'
    print('gpu')

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
        p = non_max_suppression(preds[0])
        
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]    # second output is len 3 if pt, but only 1 if exported
        
        for i, pred in enumerate(p):
            if MODE == 0:
                pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_imgs.shape)
                results = Results(orig_img=orig_imgs, boxes=pred[:, :6])
                output = results.plot()

            elif MODE == 1:
                masks = process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                if not isinstance(orig_imgs, torch.Tensor):
                    pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_imgs.shape)
                results = Results(orig_img=orig_imgs, boxes=pred[:, :6], masks=masks)
                output = results.plot()
        return output

    def stream_inference(self, frame=None):
        """Streams real-time inference on camera feed and saves results to file."""
        im = self.preprocess(frame, model, device)
        with torch.no_grad():
            preds = model(im, augment=False, visualize=False)
        results = self.postprocess(preds, im, frame)
        return results