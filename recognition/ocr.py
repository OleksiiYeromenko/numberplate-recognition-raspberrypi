import easyocr
import numpy as np


class EasyOcr():
    def __init__(self, lang = ['en'], allow_list = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', min_size=50):
        self.reader = easyocr.Reader(lang, gpu=False)
        self.allow_list = allow_list
        self.min_size = min_size

    def run(self, img):
        ocr_result = self.reader.readtext(img, allowlist = self.allow_list, min_size=self.min_size)#, paragraph="True"
        text = [x[1] for x in ocr_result]
        confid = [x[2] for x in ocr_result]
        return {'text': "".join(text), 'confid':np.round(np.mean(confid),2)}


if __name__ == "__main__":
    pass
    #TBD - add test