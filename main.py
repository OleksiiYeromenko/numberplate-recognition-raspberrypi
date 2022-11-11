# import sys
# # sys.path.append('./detection/yolov7/')

from camera.camera import PiCamera
from detection.detector import Detector
from visualization.visual import Visualize
from recognition.ocr import EasyOcr

import argparse
import time

import cv2






# camera = CAMERA()
# detect = DETECTOR()
# ocr = OCR()
# prnt = PRINT()
#
# while true:
#     img= camera.generate()
#     bbox = detect(img)
#     number = ocr(bbox)
# 	if show_result:
# 		prnt (img, bbox, number)
#
# +Unit tests - each method classa
# +ssh script to copy from git, install requirements, run tests
# +logging to file
#
#
#
#

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--log-level', type=str, default='INFO', help='logging level')
    parser.add_argument('--show-img', action='store_true', help='display results')
    parser.add_argument('--save-img', action='store_true', help='save imgs to *.jpg')
    opt = parser.parse_args()
    opt = vars(opt)
    # print(opt)



    camera = PiCamera(img_size=640, fps=36)
    detector = Detector('detection/models/25ep_best.pt', log_level='INFO')
    ocr = EasyOcr(lang = ['en'], allow_list ='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', min_size=50)

    # temp
    show_img = True
    save_img = True


    # # temp:
    # img = cv2.imread('data/test/test1.jpg')

    while True:
        img = camera.run()
        # print(f"img shape:{img.shape}")
        detect_result = detector.run(img, conf_thres=0.2)
        # print(f"Detection result:{res}")
        if detect_result['cropped_img'] is not None:
            ocr_result = ocr.run(detect_result['cropped_img'])
        else:
            ocr_result = {'text': None, 'confid': None}


        if show_img or save_img:
            visualizer = Visualize(im0=detect_result['orig_img'], file_name=detect_result['file_name'],
                                   cropped_img=detect_result['cropped_img'],
                                   bbox=detect_result['bbox'], det_conf=detect_result['det_conf'],
                                   ocr_num=ocr_result['text'], ocr_conf=ocr_result['confid'], num_check_response=None,
                                   out_img_size=(720, 1280))

            if save_img:
                pass


            if show_img:
                visualizer.show()
                key=cv2.waitKey(1) & 0xFF
                # Press 'q' key to break the loop
                if key == ord("q"):
                    break
                time.sleep(0.003)



# Add inference time in ocr, add fps