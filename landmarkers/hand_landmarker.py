import mediapipe as mp
import time
import os

class Landmarker:
    def __init__(self):
        self.landmarker = self.create_landmarker()
        self.timestamp = 0  # needed for detect_for_video

    def create_landmarker(self):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "hand_landmarker.task")

        options = HandLandmarkerOptions(
            base_options = BaseOptions(model_asset_path=model_path),
            running_mode = VisionRunningMode.VIDEO,
            num_hands = 1,
            min_hand_detection_confidence = 0.5,
            min_hand_presence_confidence = 0.5,
            min_tracking_confidence = 0.5,
        )

        return HandLandmarker.create_from_options(options)

    def detect(self, frame):
        """Runs synchronous detection and returns the result."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        result = self.landmarker.detect_for_video(
            mp_image,
            self.timestamp
        )
        self.timestamp += 1

        return result

    def close(self):
        self.landmarker.close()
