import cv2
import time
import mediapipe as mp
from landmarkers.hand_landmarker import Landmarker
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

from model import ASL_MLP
import torch

from landmarkers.face_landmarker import FaceLandmarkerWrapper

from getfromdatabase import get_text


model = ASL_MLP()
model.load_state_dict(torch.load("training/asl_mlp_model25.pth", weights_only=True))
model.eval()  # set model to evaluation mode

#dictionary for translating all values from signed (right) hand
translation_dict = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: " ",   # special case
    20: "T",
    21: "U",
    22: "V",
    23: "W",
    24: "X",
    25: "Y",
    26: "Z"
}

def draw_landmarks_on_image(rgb_image, detection_result: mp.tasks.vision.HandLandmarkerResult):
   """Courtesy of https://github.com/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb"""
   try:
      if detection_result.hand_landmarks == []:
         return rgb_image
      else:
         hand_landmarks_list = detection_result.hand_landmarks
         annotated_image = np.copy(rgb_image)

         # Loop through the detected hands to visualize.
         for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            
            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
               landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
            mp.solutions.drawing_utils.draw_landmarks(
               annotated_image,
               hand_landmarks_proto,
               mp.solutions.hands.HAND_CONNECTIONS,
               mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
               mp.solutions.drawing_styles.get_default_hand_connections_style())
         return annotated_image
   except:
      return rgb_image

def predict(model, sample):
    with torch.no_grad():
        sample = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)  # add batch dimension
        output = model(sample)
        predicted_class = torch.argmax(output, dim=1).item()
        translated_prediction = translation_dict[predicted_class]

        return translated_prediction


#move out all write_text_on_image

def predict_letter(frame, result, flipper = 1): #flipper checks for lefthanded checkbox
    "takes the current hand object extruded from the frame and draws the model prediction on the screen"
    
    try:
        if result.hand_world_landmarks == []:
            return frame, None
        else:
            #transform the landmarks to the same format as the classifier is used to
            landmarks = np.array([[flipper*lm.x, lm.y, lm.z] for lm in result.hand_world_landmarks[0]])
            landmarks -= landmarks[0]
            scale = np.linalg.norm(landmarks[9] - landmarks[0])  # scales mean bone length
            landmarks /= scale

            landmarks = landmarks.flatten() # is this right?
            prediction = predict(model, landmarks) #we should probably just return the prediction

        return frame, prediction #add the things to the frame properly
    except Exception as e:
        print("Predict error: ", e)
        return frame, None

def get_input(prediction):

    global fullstring
    global correct_string
    global colorvector
    
    if len(fullstring) <= len(correct_string) and prediction:
        fullstring += prediction

        if prediction == correct_string[len(fullstring)-1]:
            colorvector[len(fullstring)-1] = 1
        else:
            colorvector[len(fullstring)-1] = 2
    return
