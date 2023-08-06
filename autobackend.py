# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np
import torch
import torch.nn as nn

from config import weights


class AutoBackend(nn.Module):

    def __init__(self, device=torch.device('cpu'), fp16=False, fuse=True):
        """
        MultiBackend class for python inference on various platforms using Ultralytics YOLO.

        Args:
            weights (str): The path to the weights file. Default: 'yolov8n.pt'
            device (torch.device): The device to run the model on.
            fp16 (bool): If True, use half precision. Default: False
            fuse (bool): Whether to fuse the model or not. Default: True

        Supported formats and their naming conventions:
            | Format                | Suffix           |
            |-----------------------|------------------|
            | PyTorch               | *.pt             |
        """
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)

        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA

        # PyTorch
        model = attempt_load_weights(weights if isinstance(weights, list) else w, device=device, fuse=fuse)
        stride = max(int(model.stride.max()), 32)  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        model.half() if fp16 else model.float()  # https://medium.com/@fanzongshaoxing/post-training-quantization-of-tensorflow-model-to-fp16-8d66b9dfa77f
        self.model = model  # explicitly assign for to(), cpu(), cuda(), half()

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False):
        """
        Runs inference on the YOLOv8 MultiBackend model.

        Args:
            im (torch.Tensor): The image tensor to perform inference on.
            augment (bool): whether to perform data augmentation during inference, defaults to False
            visualize (bool): whether to visualize the output predictions, defaults to False

        Returns:
            (tuple): Tuple containing the raw output tensor, and processed output for visualization (if visualize=True)
        """
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        """
         Convert a numpy array to a tensor.

         Args:
             x (np.ndarray): The array to be converted.

         Returns:
             (torch.Tensor): The converted tensor
         """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x


class Ensemble(nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, visualize=False):
        """Function generates the YOLOv5 network's final layer."""
        y = [module(x, augment, visualize)[0] for module in self]
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # inference, train output


def attempt_load_weights(weights, device=None, fuse=False):
    """Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a."""
    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(w, map_location='cpu') # load ckpt
        model = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model
        if not hasattr(model, 'stride'):
            model.stride = torch.tensor([32.])

        # Append
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, 'fuse') else model.eval())  # model in eval mode

    # Return model
    if len(ensemble) == 1:
        return ensemble[-1]