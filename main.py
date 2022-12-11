import sys
sys.path.append('./detection/yolov7/')

import argparse
import cv2
import time
import threading
from camera.camera import PiCamera
from detection.detector import Detector
from visualization.visual import Visualize
from recognition.ocr import EasyOcr
from validation.utils import read_allowed_numbers
from action.lego import Action


def main(opt):
    # Arguments
    img_source = opt.img_source
    video_capture_source = opt.video_capture_source
    weights = opt.weights
    img_size = opt.img_size
    det_conf_thres = opt.det_conf_thres
    log_level = opt.log_level
    cam_rotate_180 = opt.cam_rotate_180
    show_img = opt.show_img
    save_img = opt.save_img
    save_cropped = opt.save_cropped
    save_input  = opt.save_input
    display_img = opt.display_img
    
    action_status = [None]
    lock = threading.Lock()

    # Init instances
    detector = Detector(weights, log_level=log_level)
    ocr = EasyOcr(lang=['en'], allow_list='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', min_size=50, log_level=log_level)
    allowed_numbers_list = read_allowed_numbers(sheet_id='1eVjmjribNgYRRcG9i6lcztRLdC8LiB9jgLc85BiOSbU')
    try:
        #TBD
        barrier_action = Action()
        # Start barrier action as a parallel thread
        action_thread = threading.Thread(target=barrier_action.run, args=(action_status,))
        action_thread.daemon = True
        action_thread.start()
        
    except:
        print("No Lego Build HAT.")
    


    # Case input is image
    if img_source:
        img = cv2.imread(img_source)
        # Run numberplate bbox detection:
        detect_result = detector.run(img, conf_thres=det_conf_thres)
        # Run ocr on detected number region:
        ocr_result = ocr.run(detect_result)
        # Check if detected number in allowed list (API, db, etc. request). Here for test - From excel sheet on Google drive
        if ocr_result['text'] is not None and ocr_result['text'].lower() in allowed_numbers_list:
            with lock:
                action_status[0] = 'Allowed'
        elif ocr_result['text'] is not None:
            with lock:
                action_status[0] = "Prohibited"
        else:
            with lock:
                action_status[0] = None
                

        if show_img or save_img or save_cropped or display_img:
            visualizer = Visualize(im0=detect_result['orig_img'], file_name=detect_result['file_name'],
                                   cropped_img=detect_result['cropped_img'],
                                   bbox=detect_result['bbox'], det_conf=detect_result['det_conf'],
                                   ocr_num=ocr_result['text'], ocr_conf=ocr_result['confid'],
                                   num_check_response=action_status[0],
                                   out_img_size=(720, 1280), outp_orig_img_size=720,
                                   save_jpg_qual=65, log_img_qnt_limit=10800)
            if show_img:
                visualizer.show()
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            if save_img:
                visualizer.save()
            if save_cropped:
                visualizer.save_crop()
            if save_input:
                visualizer.save_input()
            if display_img:
                visualizer.display()
#             
#         if ocr_result['text'] is not None:
#             try:
#                 #TBD
#                 barrier_action.run(action_status)
#             except:
#                 print("No Lego Build HAT.")

    # Case input is camera
    else:
        camera = PiCamera(src=video_capture_source, img_size=img_size, rotate_180=cam_rotate_180, fps=1)  #
        # Continually purging the camera's frame buffer to fix lag when processing old frame from buffer
        frame_rate = 0.6 #aprx for 1.5s per frame
        prev_time = 0
        i=0
        while True:
            i+=1
            print(f"######################### frame: {i}")
            # 
            time_elapsed = time.time() - prev_time
#             print(f"time_elapsed:{time_elapsed}")
            # Take a frame from camera
            img = camera.run()
            # Check that we are processing only frame with desired rate
            if time_elapsed > 1./frame_rate:
                prev_time = time.time()
                # Run numberplate bbox detection:
                detect_result = detector.run(img, conf_thres=det_conf_thres)
                # Run ocr on detected number region:
                ocr_result = ocr.run(detect_result)
                # Check if detected number in allowed list (API, db, etc. request). Here for test - From excel sheet on Google drive
                if ocr_result['text'] is not None and ocr_result['text'].lower() in allowed_numbers_list:
                    with lock:
                        action_status[0] = 'Allowed'
                elif ocr_result['text'] is not None:
                    with lock:
                        action_status[0] = "Prohibited"
                else:
                    with lock:
                        action_status[0] = None
                if show_img or save_img or save_cropped or display_img:
                    visualizer = Visualize(im0=detect_result['orig_img'], file_name=detect_result['file_name'],
                                           cropped_img=detect_result['cropped_img'],
                                           bbox=detect_result['bbox'], det_conf=detect_result['det_conf'],
                                           ocr_num=ocr_result['text'], ocr_conf=ocr_result['confid'],
                                           num_check_response=action_status[0],
                                           out_img_size=(720, 1280), outp_orig_img_size=720,
                                           save_jpg_qual=65, log_img_qnt_limit=10800)
                    if show_img:
                        visualizer.show()
                        # Press 'q' key to stop showing, loop will continue:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            cv2.destroyAllWindows()
                            show_img = False
                            continue
                    if save_img:
                        visualizer.save()
                    if save_cropped:
                        visualizer.save_crop()
                    if save_input:
                        visualizer.save_input()
                    if display_img:
                        visualizer.display()
#                     
#                 if ocr_result['text'] is not None:               
#                     try:
#                         #TBD
#                         barrier_action.run(action_status)
#                     except:
#                         print("No Lego Build HAT.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-source', type=str, default=None, help='run with image input, not camera')
    parser.add_argument('--video_capture_source', type=str, default=None, help='path to videofile, None - Webcam')
    parser.add_argument('--weights', type=str, default='detection/models/25ep_best.pt', help='inference size (pixels)')
    parser.add_argument('--img-size', type=tuple, default=(1280,720), help='camera image size (pixels)')
    parser.add_argument('--det-conf-thres', type=float, default=0.25, help='object det confidence threshold')
    parser.add_argument('--log-level', type=str, choices=('INFO', 'DEBUG'), default='INFO', help='logging level')
    parser.add_argument('--cam-rotate-180', action='store_true', help='flip camera image 180')
    parser.add_argument('--show-img', action='store_true', help='display results')
    parser.add_argument('--save-img', action='store_true', help='save imgs to *.jpg')
    parser.add_argument('--save-cropped', action='store_true', help='save cropped imgs to *.jpg')
    parser.add_argument('--save-input', action='store_true', help='save input imgs to *.jpg')
    parser.add_argument('--display-img', action='store_true', help='display imgs to e-ink')
    opt = parser.parse_args()

    # Dev testing
    opt.cam_rotate_180 = True
    
    opt.show_img = True
#     opt.display_img = True
#     opt.save_img = True
#     opt.save_cropped = True
#     opt.save_input = True
#     opt.log_level = 'DEBUG'
#     opt.img_source = 'data/test/test1.jpg'
#     opt.video_capture_source = './data/test/videorec_28112022194352.mp4'

    main(opt)
    
    

    