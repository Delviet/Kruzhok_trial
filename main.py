import sys

import numpy as np
import torch
import cv2

from models.experimental import attempt_load

from utils.general import non_max_suppression
from utils.datasets import letterbox
from utils.torch_utils import select_device

import typer


def main(path: str):
    device = select_device('')
    model = attempt_load('weights.pt', device)
    
    img_size = 640
    conf_thres = 0.5
    iou_thres = 0.5

    img0 = cv2.imread(path)

    # Padded resize
    img = letterbox(img0, new_shape=img_size)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float()

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    img /= 255.0

    pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    for i, det in enumerate(pred):
        if det is not None:
            for *xyxy, conf, cls in reversed(det):
                if cls == 0:
                    print("kruzhok")
                    sys.exit(0)


if __name__ == '__main__':
    typer.run(main)
