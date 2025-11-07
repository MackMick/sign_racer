import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np
import math

class FaceLandmarkerWrapper:
    def __init__(self, model_path="face_landmarker.task"):
        self.model_path = model_path
        self.timestamp = 0
        self.landmarker = self._init_landmarker()
        self.open = True #are the eyes open?

    def _init_landmarker(self):
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO
        )

        return FaceLandmarker.create_from_options(options)

    def detect(self, frame):
        """Run landmark detection on a single frame and return landmarks."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = self.landmarker.detect_for_video(mp_image, self.timestamp)
        self.timestamp += 1

        if not result.face_landmarks:
            return None

        # Convert to NormalizedLandmarkList for easy downstream use/drawing
        face_landmarks = []
        for lm_list in result.face_landmarks:
            proto = landmark_pb2.NormalizedLandmarkList(
                landmark=[
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                    for lm in lm_list
                ]
            )
            face_landmarks.append(proto)

        return face_landmarks


    def take_input(self, landmarks):
        #seperate out distance calculations to a helper function instead, also the translating?
        flag = False
        LEFT_EYE_IMPORTANT = {386,374,362,263}#upper,lower,left,right
        RIGHT_EYE_IMPORTANT = {159,145,33,133} #upper,lower,left,right

        threshold = 0.25

        #left eye
        left_upper = landmarks[0].landmark[386]
        left_lower = landmarks[0].landmark[374]
        left_left = landmarks[0].landmark[362]
        left_right = landmarks[0].landmark[263]

        left_up_down_dist = np.linalg.norm(np.array([left_upper.x, left_upper.y]) - np.array([left_lower.x, left_lower.y]))
        left_left_right_dist = np.linalg.norm(np.array([left_left.x, left_left.y]) - np.array([left_right.x, left_right.y]))

        left_ratio = left_up_down_dist/left_left_right_dist
        
        right_upper = landmarks[0].landmark[159]
        right_lower = landmarks[0].landmark[145]
        right_left  = landmarks[0].landmark[33]
        right_right = landmarks[0].landmark[133]

        right_up_down_dist = np.linalg.norm(
            np.array([right_upper.x, right_upper.y]) - np.array([right_lower.x, right_lower.y])
        )

        right_left_right_dist = np.linalg.norm(
            np.array([right_left.x, right_left.y]) - np.array([right_right.x, right_right.y])
        )

        right_ratio = right_up_down_dist / right_left_right_dist

        ratio = (right_ratio + left_ratio)/2
        
        if ratio < threshold and self.open == True:
            self.open = False
            return True
        elif ratio > threshold:
            self.open = True

        return flag


    def draw(self, frame, landmarks):
        """Draw face mesh landmarks on the frame."""
        mp_drawing = mp.solutions.drawing_utils
        mp_styles = mp.solutions.drawing_styles
        mp_mesh = mp.solutions.face_mesh

        LEFT_EYE_INDEXES = {33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246}
        RIGHT_EYE_INDEXES = {263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466}
        EYE_INDEXES = LEFT_EYE_INDEXES | RIGHT_EYE_INDEXES


        for lm_proto in landmarks:
            eyes = landmark_pb2.NormalizedLandmarkList(landmark=[lm_proto.landmark[i] for i in EYE_INDEXES])

            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=eyes,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                connections=None
            )

        return frame
    
