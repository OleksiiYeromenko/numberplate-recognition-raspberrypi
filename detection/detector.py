from pathlib import Path
import sys
import torch
import cv2
import numpy as np
from numpy import random
import time
import logging
from logging.handlers import RotatingFileHandler
from detection.utils import attempt_load, TracedModel, LoadImage, non_max_suppression, scale_coords, xyxy2xywh, \
    save_crop


class Detector():
    """Numberplate detector based on yolov7
     Input: path to a file or cv2 BGR format
     .run(inp_image, conf_thres) returns: Filename, Cropped image; Original image, Bounding box"""

    def __init__(self, model_weights, img_size=640, device='cpu', half=False, trace=True, log_level='INFO', log_dir = './logs/'):
        # Initialize
        self.model_weights = model_weights
        self.img_size = img_size
        self.device = torch.device(device)
        self.half = half  # half = device.type != 'cpu'  # half precision only supported on CUDA
        self.trace = trace  # Convert model to Traced-model
        self.log_level = log_level
        if self.log_level:
            self.num_log_level = getattr(logging, self.log_level.upper(), 20) ##Translate the log_level input string to one of the accepted values of the logging module, if no 20 - INFO
            self.log_dir = log_dir

            log_formatter = logging.Formatter("%(asctime)s %(message)s")
            logFile = self.log_dir + 'detection.log'
            my_handler = RotatingFileHandler(logFile, mode='a', maxBytes=25 * 1024 * 1024,
                                             backupCount=10, encoding='utf-8', delay=False)
            my_handler.setFormatter(log_formatter)
            my_handler.setLevel(self.num_log_level)
            self.logger = logging.getLogger(__name__)  # logging.getLogger(__name__)  .getLogger('root')
            self.logger.setLevel(self.num_log_level)
            self.logger.addHandler(my_handler)

        # Add path to yolo model as whenever load('weights.pt') is called, pytorch looks for model config in path enviornment variable (models/yolo)
        yolo_folder_dir = str(Path(__file__).parent.absolute()) +"\yolov7" #  models folder path
        sys.path.insert(0, yolo_folder_dir)

        # Load model
        self.model = attempt_load(self.model_weights, map_location=self.device)  # load FP32 model
        # stride = int(self.model.stride.max())  # model stride
        # imgsz = check_img_size(self.img_size, s=stride)  # check img_size

        # Convert model to Traced-model
        if self.trace:
            self.model = TracedModel(self.model, self.device, self.img_size)
        # if half:
        #     model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        if len(self.names) > 1:
            self.colors = [[0, 255, 127]] + [[random.randint(0, 255) for _ in range(3)] for _ in self.names[1:]]
        else:
            self.colors = [[0, 255, 127]]

        sys.path.remove(yolo_folder_dir)

    def run(self, inp_image, conf_thres=0.25):
        # Run Inference

        # Load data
        dataset = LoadImage(inp_image, device=self.device, half=self.half)
        t0 = time.time()
        self.file_name, self.img, self.im0 = dataset.preprocess()

        # Inference
        t1 = time.time()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            self.pred = self.model(self.img)[0]
        t2 = time.time()

        # Apply NMS
        self.pred = non_max_suppression(self.pred, conf_thres=conf_thres)
        t3 = time.time()

        # Process detections
        bbox = None  # bounding boxe of detected object with max conf
        cropped_img = None  # cropped detected object with max conf
        det_conf = None  # confidence level for detected object with max conf

        self.det = self.pred[0]  # pred[0] - NMX suppr returns list with 1 tensor per image;

        if len(self.det):
            # Rescale boxes from img_size to im0 size
            self.det[:, :4] = scale_coords(self.img.shape[2:], self.det[:, :4], self.im0.shape).round()

            # Print results
            print_strng = ""
            for c in self.det[:, -1].unique():
                n = (self.det[:, -1] == c).sum()  # detections per class
                print_strng += f"{n} {self.names[int(c)]}{'s' * (n > 1)}"  # add to string

            # Print time (inference + NMS)
            print(
                f'{print_strng} detected. ({(1E3 * (t1 - t0)):.1f}ms)-Load data, ({(1E3 * (t2 - t1)):.1f}ms)-Inference, ({(1E3 * (t3 - t2)):.1f}ms)-NMS')

            # Write results to file if debug mode
            if self.log_level:
                self.logger.debug(
                    f'{self.file_name} {print_strng} detected. ({(1E3 * (t1 - t0)):.1f}ms)-Load data, ({(1E3 * (t2 - t1)):.1f}ms)-Inference, ({(1E3 * (t3 - t2)):.1f}ms)-NMS')
                if self.logger.getEffectiveLevel() == 10:  # level 10 = debug
                    gn = torch.tensor(self.im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    for *xyxy, conf, cls in reversed(self.det):
                        # save detections with bbox in xywh format
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (int(cls), np.round(conf, 3), *xywh)  # label format
                        self.logger.debug(f"{self.file_name} {('%g ' * len(line)).rstrip() % line}")

            # Find detection with max confidence:
            indx = self.pred[0].argmax(0)[
                4]  # pred[0] - NMX suppr returns list with 1 tensor per image; argmax(0)[4] - conf has indx 4 in [x1,y1,x2,y2,conf,cls]
            max_det = self.pred[0][indx]
            # Collect detected bounding boxe and corresponding cropped img
            bbox = max_det[:4]
            cropped_img = save_crop(max_det[:4], self.im0)
            cropped_img = cropped_img[:, :, ::-1] # # BGR to RGB
            det_conf = max_det[4:5]

        print(f'Detection total time: {time.time() - t0:.3f}s')
        return {'file_name': self.file_name, 'orig_img': self.im0, 'cropped_img': cropped_img, 'bbox': bbox,
                'det_conf': det_conf}


if __name__ == "__main__":
    weights = '../detection/models/25ep_best.pt'
    img = cv2.imread('../data/test/test1.jpg')
    detector = Detector(weights, log_level=None)
    detect_result = detector.run(img, conf_thres=0.35)

    print(detect_result.get('file_name'))
    img = detect_result.get('orig_img')
    print(img.shape)
    print(detect_result.get('cropped_img').shape)
    bbox = detect_result.get('bbox')
    print(bbox)
    conf = detect_result.get('det_conf')
    print(conf)

    # plot img
    if bbox is not None:
        c1, c2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
        color = [random.randint(0, 255) for _ in range(3)]
        cv2.rectangle(img, c1, c2, color=color,thickness=3, lineType=cv2.LINE_AA)
    if conf is not None:
        cv2.putText(img, f'{conf.item():.2f}', (c1[0], c1[1] - 2), 0, 1, color, thickness=3, lineType=cv2.LINE_AA)
    cv2.imshow('image', img)
    cv2.waitKey()
