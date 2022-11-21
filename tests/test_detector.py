from detection.detector import Detector
import cv2
import numpy as np


def test_run() -> None:
    weights = './detection/models/25ep_best.pt'
    img = cv2.imread('./data/test/test1.jpg')

    detector = Detector(weights, log_level=None)
    detect_result = detector.run(img, conf_thres=0.25)

    assert detect_result.get('orig_img').shape == (742, 1200, 3)
    assert detect_result.get('cropped_img').shape == (51, 244, 3)
    assert detect_result.get('bbox').tolist() == [459., 337., 703., 388.]
    assert np.round(detect_result.get('det_conf').tolist(),2) == [0.91]

