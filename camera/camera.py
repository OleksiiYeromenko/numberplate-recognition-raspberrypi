import cv2

class PiCamera():
    def __init__(self, src=0, img_size=640, fps=36, rotate_180=False):
        self.img_size = img_size
        self.fps = fps
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_size)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_size)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.rotate_180 = rotate_180

    def run(self):
        # read frame
        ret, image = self.cap.read()
        if self.rotate_180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        if not ret:
            raise RuntimeError("failed to read frame")
        return image  # in BGR


#
# ####
# cap.release()
# cv2.destroyAllWindows()
# ###
#
# import numpy as np
# import cv2
#
# cap = cv2.VideoCapture('intro.mp4')
# while (cap.isOpened()):
#
#     ret, frame = cap.read()
#     # cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
#     # cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
#
#     if ret:
#         cv2.imshow("Image", frame)
#     else:
#         print('no video')
#         cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#         continue
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#
#
#
#








if __name__ == "__main__":
    pass
    #TBD - add test