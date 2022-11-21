from recognition.ocr import EasyOcr
import cv2

def test_ocr_run() -> None:
    ocr = EasyOcr(lang=['en'], allow_list='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', min_size=50, log_level=None)

    img = cv2.imread('../data/test/wp57yws.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detect_result_dict ={'cropped_img': img}

    res = ocr.run(detect_result_dict)
    numbr = res.get('text')
    confid = res.get('confid')
    assert numbr == 'WP57YHS'
    assert confid == 0.44







