import numpy as np
import cv2
import logging
from logging.handlers import RotatingFileHandler

import sys
import os
libdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lib')
if os.path.exists(libdir):
    sys.path.append(libdir)
import epd2in7
import time
from PIL import Image


from detection.utils import plot_one_box



class Visualize():
    def __init__(self, im0, file_name, cropped_img=None, bbox=None, det_conf=None, ocr_num=None, ocr_conf=None, num_check_response=None, out_img_size=(720,1280), orig_img_size = 640, log_level='INFO'):
        self.im0 = im0
        self.file_name = file_name
        self.cropped_img = cropped_img
        self.bbox = bbox
        self.det_conf = det_conf
        self.ocr_num = ocr_num
        self.ocr_conf = ocr_conf
        self.num_check_response = num_check_response
        self.out_img_size = out_img_size
        self.num_log_level = getattr(logging, log_level.upper(), 20) ##Translate the log_level input string to one of the accepted values of the logging module, if no 20 - INFO
        # Set logger
        log_formatter = logging.Formatter("%(asctime)s %(message)s")
        logFile = './detection/logs/detection.log'
        my_handler = RotatingFileHandler(logFile, mode='a', maxBytes=25 * 1024 * 1024,
                                         backupCount=30, encoding='utf-8', delay=False)
        my_handler.setFormatter(log_formatter)
        my_handler.setLevel(self.num_log_level)
        self.logger = logging.getLogger('root')  # logging.getLogger(__name__)
        self.logger.setLevel(self.num_log_level)
        self.logger.addHandler(my_handler)
        

        # Create blank image
        h, w = self.out_img_size
        self.img = np.zeros((h, w, 3), np.uint8)
        self.img[:, :] = (255, 255, 255)

        # Draw bounding box on top the image
        if (self.bbox is not None) and (self.det_conf is not None):
            label = f'{self.det_conf.item():.2f}'
            color = [0, 255, 127]
            plot_one_box(self.bbox, self.im0, label=label, color=color, line_thickness=3)

        # Resize img width to fit the plot, keep origin aspect ratio
        r = orig_img_size / self.im0.shape[1]
        dim = (orig_img_size, int(self.im0.shape[0] * r))
        self.im0 = cv2.resize(self.im0, dim, interpolation=cv2.INTER_AREA)
        im0_h, im0_w = self.im0.shape[:2]

        # Add original full image
        im0_offset = 5
        self.img[im0_offset:im0_h + im0_offset, im0_offset:im0_w + im0_offset] = self.im0


        if self.cropped_img is not None:
            # Resize cropped img
            target_width = int((w - (im0_w + im0_offset)) / 3)
            r = target_width / self.cropped_img.shape[1]
            dim = (target_width, int(self.cropped_img.shape[0] * r))
            self.cropped_img = cv2.resize(self.cropped_img, dim, interpolation=cv2.INTER_AREA)
            crop_h, crop_w = self.cropped_img.shape[:2]
            # Add cropped img
            crop_h_offset = int(h/7)
            crop_w_offset = im0_w + im0_offset + int((w - (im0_w + im0_offset) - crop_w) / 2)
            self.img[crop_h_offset:crop_h + crop_h_offset, crop_w_offset:crop_w + crop_w_offset] = self.cropped_img


        if self.ocr_num is not None:
            label = f"{self.ocr_num} ({self.ocr_conf})"
            print(label)
            t_thickn = 2  # text font thickness in px
            font = cv2.FONT_HERSHEY_SIMPLEX  # font
            fontScale = 1.05
            # calculate position
            text_size = cv2.getTextSize(label, font, fontScale=fontScale, thickness=t_thickn)[0]
            w_center = int((im0_w + im0_offset + w)/2)
            x1 = int(w_center - text_size[0]/2)
            y1 = int(2 * h /7)
            org = (x1,y1)  # position
            # Plot text on img
            cv2.putText(self.img, label, org, font, fontScale,  color=(0, 0, 0), thickness=t_thickn, lineType=cv2.LINE_AA)


    def show(self):
        # Show the image
        cv2.imshow('image', self.img)
        # Wait for a key
        # cv2.waitKey(0)
        # # Distroy all the window open
        # cv2.destroyAllWindows()


    def save_img(self):
        pass


    # Didplay Rapberry Pi system information and utilization on e-ink display 176*264.
    def display(self):
        # resize image
        #TBD - take only crop and number part of an image
        disp_img = self.img[:360, 720:] 
        
        Himage = cv2.resize(disp_img, (epd2in7.EPD_HEIGHT, epd2in7.EPD_WIDTH), interpolation=cv2.INTER_AREA)
        # convert to PIL format
        Himage = Image.fromarray(Himage)
        tic = time.perf_counter()
        epd = epd2in7.EPD() # get the display
        epd.init()           # initialize the display
    #     logging.info("Clear...")    # logging.infos to console, not the display, for debugging
        epd.Clear(0xFF)      # clear the display
        toc = time.perf_counter()
        self.logger.info(f"Init, clean display - {toc - tic:0.4f} seconds")
        print(f"Init, clean display - {toc - tic:0.4f} seconds")
            
        tic = time.perf_counter()
        epd.display(epd.getbuffer(Himage))
        toc = time.perf_counter()
        self.logger.info(f"Display image - {toc - tic:0.4f} seconds")
        print(f"Display image - {toc - tic:0.4f} seconds")
        epd.sleep() # Power off display
        

        
        
    
    
    
    
    