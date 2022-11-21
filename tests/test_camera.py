from camera.camera import PiCamera


def test_camera_run() -> None:
    camera = PiCamera(img_size=640)
    im = camera.run()
    assert im.shape == (480, 640, 3)


