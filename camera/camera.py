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




if __name__ == "__main__":
    camera = PiCamera()

    while True:
        # Take a frame from camera
        img = camera.run()
        print(img.shape)

        # Show the image
        cv2.imshow('image', img)

        # Press 'q' key to stop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            break

    camera.cap.release()
    cv2.destroyAllWindows()
