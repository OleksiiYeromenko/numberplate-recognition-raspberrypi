import numpy as np
import cv2
import torch
import sys
import os
libdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lib')
if os.path.exists(libdir):
    sys.path.append(libdir)
try:
    import epd2in7
except:
    print("Cannot find sysfs_software_spi.so - no Serial Peripheral Interface.")

from pathlib import Path
import time
from PIL import Image
from visualization.utils import plot_one_box, check_imshow


class Visualize():
    def __init__(self, im0, file_name, cropped_img=None, bbox=None, det_conf=None, ocr_num=None, ocr_conf=None, num_check_response=None, out_img_size=(720,1280), outp_orig_img_size = 640, log_dir ='./logs/', save_jpg_qual = 65, log_img_qnt_limit = 10800):
        self.im0 = im0
        self.input_img = im0.copy()
        self.file_name = file_name
        self.cropped_img = cropped_img
        self.bbox = bbox
        self.det_conf = det_conf
        self.ocr_num = ocr_num
        self.ocr_conf = ocr_conf
        self.num_check_response = num_check_response
        self.out_img_size = out_img_size
        self.save_jpg_qual = save_jpg_qual
        self.log_dir = log_dir
        self.imgs_log_dir = self.log_dir + 'imgs/'
        os.makedirs(os.path.dirname(self.imgs_log_dir), exist_ok=True)
        self.crop_imgs_log_dir = self.log_dir + 'imgs/crop/'
        os.makedirs(os.path.dirname(self.crop_imgs_log_dir), exist_ok=True)
        self.orig_imgs_log_dir = self.log_dir + 'imgs/inp/'
        os.makedirs(os.path.dirname(self.orig_imgs_log_dir), exist_ok=True)
        self.log_img_qnt_limit = log_img_qnt_limit

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


        h0, w0 = im0.shape[:2]
        aspect = w0 / h0
        if aspect > 1:  # horizontal image
            new_w = outp_orig_img_size
            new_h = np.round(new_w / aspect).astype(int)
        elif aspect < 1:  # vertical image
            new_h = outp_orig_img_size
            new_w = np.round(new_h * aspect).astype(int)
        else:  # square image
            new_h, new_w = outp_orig_img_size, outp_orig_img_size
        # r = outp_orig_img_size / self.im0.shape[1]
        # dim = (outp_orig_img_size, int(self.im0.shape[0] * r))
        self.im0 = cv2.resize(self.im0, (new_w, new_h), interpolation=cv2.INTER_AREA)
        im0_h, im0_w = self.im0.shape[:2]

        # Add original full image
        im0_offset = 0
        self.img[im0_offset:im0_h + im0_offset, im0_offset:im0_w + im0_offset] = self.im0

        # Add cropped image with detected number bbox
        if self.cropped_img is not None:
            # Resize cropped img
            target_width = int((w - (im0_w + im0_offset)) / 3)
            r = target_width / self.cropped_img.shape[1]
            dim = (target_width, int(self.cropped_img.shape[0] * r))
            self.cropped_img = cv2.resize(self.cropped_img, dim, interpolation=cv2.INTER_AREA)
            crop_h, crop_w = self.cropped_img.shape[:2]
            # Add cropped img
            crop_h_y1 = int(h/7)
            crop_w_x1 = im0_w + im0_offset + int((w - (im0_w + im0_offset) - crop_w) / 2)
            self.img[crop_h_y1:crop_h + crop_h_y1, crop_w_x1:crop_w + crop_w_x1] = self.cropped_img
            # Add `_det` to filename
            self.file_name = Path(self.file_name).stem + "_det" + Path(self.file_name).suffix

        # Add ocr recognized number
        if self.ocr_num is not None:
            label = f"{self.ocr_num} ({self.ocr_conf})"
            t_thickn = 2  # text font thickness in px
            font = cv2.FONT_HERSHEY_SIMPLEX  # font
            fontScale = 1.05
            # calculate position
            text_size = cv2.getTextSize(label, font, fontScale=fontScale, thickness=t_thickn)[0]
            w_center = int((im0_w + im0_offset + w)/2)
            ocr_w_x1 = int(w_center - text_size[0]/2)
            ocr_h_y1 = int(crop_h_y1 + crop_h + 55)
            org = (ocr_w_x1, ocr_h_y1)  # position
            # Plot text on img
            cv2.putText(self.img, label, org, font, fontScale,  color=(0, 0, 0), thickness=t_thickn, lineType=cv2.LINE_AA)

        # Add number check response in allowed list
        if self.num_check_response == 'Allowed':
            label = "-=Allowed=-"
            fontColor = (0,255,0)
        else:
            label = "-=Prohibited!=-"
            fontColor = (0,0,255)
        t_thickn = 2  # text font thickness in px
        font = cv2.FONT_HERSHEY_SIMPLEX  # font
        fontScale = 1.05
        # calculate position
        text_size = cv2.getTextSize(label, font, fontScale=fontScale, thickness=t_thickn)[0]
        w_center = int((im0_w + im0_offset + w) / 2)
        response_w_x1 = int(w_center - text_size[0] / 2)
        response_h_y1 = int(h*3/7) #TBD
        org = (response_w_x1, response_h_y1)  # position
        # Plot text on img
        cv2.putText(self.img, label, org, font, fontScale, color=fontColor, thickness=t_thickn, lineType=cv2.LINE_AA)

    def show(self):
        # Show the image
        cv2.imshow('image', self.img)

    def save(self):
        # Remove oldest file if reach quantity limit
        if self.get_dir_file_quantity(self.imgs_log_dir) > self.log_img_qnt_limit:
            oldest_file = sorted([self.imgs_log_dir+f for f in os.listdir(self.imgs_log_dir)])[
                0]  # , key=os.path.getctime
            os.remove(oldest_file)
        # Write compressed jpeg with results
        cv2.imwrite(f"{self.imgs_log_dir}{self.file_name}", self.img, [int(cv2.IMWRITE_JPEG_QUALITY), self.save_jpg_qual])
        # TBD Write in byte string format

    def save_input(self):
        if self.input_img is not None:
            # Remove oldest file if reach quantity limit
            if self.get_dir_file_quantity(self.orig_imgs_log_dir) > self.log_img_qnt_limit:
                oldest_file = sorted([self.orig_imgs_log_dir+f for f in os.listdir(self.orig_imgs_log_dir)])[
                    0]  # , key=os.path.getctime
                os.remove(oldest_file)
            # Write compressed jpeg with results
            cv2.imwrite(f"{self.orig_imgs_log_dir}orig_inp_{self.file_name}", self.input_img)
            # TBD Write in byte string format

    def save_crop(self):
        if self.cropped_img is not None:
            # Remove oldest file if reach quantity limit
            if self.get_dir_file_quantity(self.crop_imgs_log_dir) > self.log_img_qnt_limit:
                oldest_file = sorted([self.crop_imgs_log_dir+f for f in os.listdir(self.crop_imgs_log_dir)])[
                    0]  # , key=os.path.getctime
                os.remove(oldest_file)
            # Write compressed jpeg with results
            cv2.imwrite(f"{self.crop_imgs_log_dir}crop_{self.file_name}", self.cropped_img)
            # TBD Write in byte string format

    # Display img on e-ink display 176*264.
    def display(self):
        # Create blank image
        disp_img = np.zeros((epd2in7.EPD_WIDTH, epd2in7.EPD_HEIGHT,3), np.uint8)
        disp_img[:, :] = (255, 255, 255)
        
        if self.cropped_img is not None:
            # Add cropped number
            crop_resized = cv2.resize(self.cropped_img, (epd2in7.EPD_HEIGHT-4, 85), interpolation=cv2.INTER_AREA)
            crop_resized_h, crop_resized_w = crop_resized.shape[:2]
            crop_w_x1 = int(epd2in7.EPD_HEIGHT/2 - crop_resized_w/2)
            disp_img[2:crop_resized_h+2, crop_w_x1:crop_resized_w+crop_w_x1] = crop_resized
        
        if self.ocr_num is not None:
            # Add recognized label
            label = f"{self.ocr_num}({self.ocr_conf})"
            t_thickn = 2  # text font thickness in px
            font = cv2.FONT_HERSHEY_SIMPLEX  # font
            fontScale = 0.8
            text_size = cv2.getTextSize(label, font, fontScale=fontScale, thickness=t_thickn)[0]
            ocr_w_x1 = int(epd2in7.EPD_HEIGHT / 2 - text_size[0] / 2)
            ocr_h_y1 = int(crop_resized_h/2 +2 + epd2in7.EPD_WIDTH/2)
            # Plot text on img
            cv2.putText(disp_img, label, (ocr_w_x1, ocr_h_y1), font, fontScale, color=(0, 0, 0), thickness=t_thickn, lineType=cv2.LINE_AA)
        Himage = cv2.resize(disp_img, (epd2in7.EPD_HEIGHT, epd2in7.EPD_WIDTH), interpolation=cv2.INTER_AREA)
        print(f"###Himage:{Himage.shape}")
        # convert to PIL format
        Himage = Image.fromarray(Himage)
        tic = time.perf_counter()
        epd = epd2in7.EPD() # get the display
        epd.init()           # initialize the display
    #     logging.info("Clear...")    # logging.infos to console, not the display, for debugging
        epd.Clear(0xFF)      # clear the display
        toc = time.perf_counter()
        print(f"Init, clean display - {toc - tic:0.4f} seconds")
            
        tic = time.perf_counter()
        epd.display(epd.getbuffer(Himage))
        toc = time.perf_counter()
        print(f"Display image - {toc - tic:0.4f} seconds")
        epd.sleep() # Power off display
        
    @staticmethod
    def get_dir_file_quantity(dir_path):
        list_of_files = os.listdir(dir_path)
        return len(list_of_files)

        
    


if __name__ == "__main__":

    show_img =True
    display_img = False
    save_img = True
    save_cropped = True
    save_inpu = True

    img = cv2.imread('../data/test/test1.jpg')
    bbox = torch.tensor([459., 337., 703., 388.])
    crop = cv2.imread('../data/test/plates/SN66XMZ.png')
    det_conf = torch.tensor(0.91)
    ocr_num = 'SN66XMZ'
    ocr_conf = '0.99'
    file_name = 'test.jpg'

    if show_img or display_img or save_img or save_cropped:
        visualizer = Visualize(im0=img, file_name=file_name,
                               cropped_img=crop,
                               bbox=bbox, det_conf=det_conf,
                               ocr_num=ocr_num, ocr_conf=ocr_conf,
                               num_check_response='Allowed',
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

    
    