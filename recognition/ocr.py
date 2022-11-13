import easyocr
import numpy as np
import time
import logging
from logging.handlers import RotatingFileHandler

class EasyOcr():
    def __init__(self, lang = ['en'], allow_list = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', min_size=50, log_level='INFO', log_dir = './logs/'):
        self.reader = easyocr.Reader(lang, gpu=False)
        self.allow_list = allow_list
        self.min_size = min_size
        self.num_log_level = getattr(logging, log_level.upper(),
                                     20)  ##Translate the log_level input string to one of the accepted values of the logging module, if no 20 - INFO
        self.log_dir = log_dir

        # Set logger
        log_formatter = logging.Formatter("%(asctime)s %(message)s")
        logFile = self.log_dir + 'ocr.log'
        my_handler = RotatingFileHandler(logFile, mode='a', maxBytes=25 * 1024 * 1024,
                                         backupCount=10, encoding='utf-8', delay=False)
        my_handler.setFormatter(log_formatter)
        my_handler.setLevel(self.num_log_level)
        self.logger = logging.getLogger(__name__)  # logging.getLogger(__name__) .getLogger('root')
        self.logger.setLevel(self.num_log_level)
        self.logger.addHandler(my_handler)

    def run(self, detect_result_dict):
        if detect_result_dict['cropped_img'] is not None:
            t0 = time.time()
            img = detect_result_dict['cropped_img']
            file_name = detect_result_dict['file_name']
            ocr_result = self.reader.readtext(img, allowlist = self.allow_list, min_size=self.min_size)#, paragraph="True"
            text = [x[1] for x in ocr_result]
            confid = [x[2] for x in ocr_result]
            text = "".join(text) if len(text) > 0 else None
            confid = np.round(np.mean(confid), 2) if len(confid) > 0 else None   #TBD
            t1 = time.time()
            print(f'Recognized number: {text}, conf.:{confid}.\nOCR total time: {(t1 - t0):.3f}s')

            # Write results to file if debug mode
            self.logger.debug(f'{file_name} Recognized number: {text}, conf.:{confid}, OCR total time: {(t1 - t0):.3f}s.')

            return {'text': text, 'confid': confid}
        else:
            return {'text': None, 'confid': None}


if __name__ == "__main__":
    pass
    #TBD - add test