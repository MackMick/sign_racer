import cv2, os

def open_camera_prefer_path():
    # If you know the right node on this machine, use it directly
    for dev_path in ("/dev/video0", "/dev/video2", "/dev/video1", "/dev/video3"):
        if os.path.exists(dev_path):
            cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
            if cap.isOpened():
                # Optional: force MJPG/size/FPS for color streams
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
                ok, frame = cap.read()
                if ok and frame is not None and frame.ndim == 3 and frame.shape[2] == 3:
                    print(f"✅ Using device {dev_path}")
                    return cap
                cap.release()
    return None
