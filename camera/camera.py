import cv2
import time


class PiCamera():
    def __init__(self, src=None, img_size=(640,480), fps=36, rotate_180=False):
        self.img_size = img_size
        self.fps = fps
        if src is None:
            src = 0
        self.cap = cv2.VideoCapture(src)
        
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#         self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_size[1])
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
    camera = PiCamera(rotate_180=True)  #src='../data/test/videorec_28112022194352.mp4'
    
    # continually purging the camera's frame buffer to fix lag when processing old frame from buffer
    frame_rate = 10
    prev_time = 0
    while True:
        time_elapsed = time.time() - prev_time
        img = camera.run()

        if time_elapsed > 1./frame_rate:
            prev_time = time.time()


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




        
#         
#     while True:
#         # Take a frame from camera
#         img = camera.run()
#         print(img.shape)
# 
#         # Show the image
#         cv2.imshow('image', img)
# 
#         # Press 'q' key to stop
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord("q"):
#             cv2.destroyAllWindows()
#             break
# 
#     camera.cap.release()
#     cv2.destroyAllWindows()
