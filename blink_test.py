import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import cv2

class FaceLandmarkerWrapper:
    def __init__(self, model_path="face_landmarker.task"):
        self.model_path = model_path
        self.timestamp = 0
        self.landmarker = self._init_landmarker()

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
        flag = False


        return flag


    def draw(self, frame, landmarks):
        """Draw face mesh landmarks on the frame."""
        mp_drawing = mp.solutions.drawing_utils
        mp_styles = mp.solutions.drawing_styles
        mp_mesh = mp.solutions.face_mesh

        for lm_proto in landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=lm_proto,
                connections=mp_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
            )



        return frame
    
