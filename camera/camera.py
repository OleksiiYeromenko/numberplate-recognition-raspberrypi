import cv2

class PiCamera():
    def __init__(self, img_size=640, fps=36):
        self.img_size = img_size
        self.fps = fps
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_size)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_size)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    def run(self):
        # read frame
        ret, image = self.cap.read()
        if not ret:
            raise RuntimeError("failed to read frame")
        return image  # in BGR




if __name__ == "__main__":
    pass
    #TBD - add test