import cv2
import time
import mediapipe as mp
from landmarker_class import Landmarker
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

from model import ASL_MLP
import torch

from blink_test import FaceLandmarkerWrapper

from getfromdatabase import get_text

correct_string = ""
current_pos = 0
colorvector = [] # makes empty array -> 
#0 -> not yet seen, 1 -> correct, 2 -> wrong
#we should really make this a real lookup thing instead of just arbitrarily ascribing colors to numbers

fullstring = ""

skinny_set = {"I"}

adjusted_placements = [20] # list to make sure all letters are placed in a good manner -> distinguish between thin and wide letters

model = ASL_MLP()
model.load_state_dict(torch.load("asl_mlp_model25.pth", weights_only=True))
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

def write_text_on_image(frame, text=""):

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (250,250)
    fontScale              = 2
    fontColor              = (255,255,255)
    thickness              = 3
    lineType               = 2

    global fullstring
    global correct_string
    global adjusted_placements

    if text != "":
        if text == " ":
            text = "(SPACE)"
        cv2.putText(frame,text, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)
    
    #TO ADD: check here if it is the correct letter -> different color if wrong
    right_color = (0,255,0) #green
    wrong_color = (0,0,255) #red
    current_color = (255,255,0) #blue
    coming_color = (255,0,0) #

    textPlacement = (10, 450) #(0,0) = topleft, (x,y)
    
    accumulated_pos = 0
    for pos, char in enumerate(correct_string):
        accumulated_pos += adjusted_placements[pos]
        adjusted_placement = (textPlacement[0] + accumulated_pos, textPlacement[1])

        if pos == len(fullstring):
            color = current_color
        elif colorvector[pos] == 0:
            color = coming_color
        elif colorvector[pos] == 1:
            color = right_color
        elif colorvector[pos] == 2:
            color = wrong_color
        else:
            print("Color Exception")
        
        if char == " ":
            char = "_"
        
        cv2.putText(frame,char, 
            adjusted_placement, 
            font, 
            fontScale,
            color,
            thickness,
            lineType)
        
    if len(fullstring) == len(correct_string):
        accuracy = colorvector.count(1)/len(colorvector) * 100
        resultstext = f"your final accuracy was: {accuracy}%"
        resultsplacement = (50,50)
        resultsscale = 1
        resultcolor = (0,255,255)
        
        cv2.putText(frame,resultstext, 
            resultsplacement, 
            font, 
            resultsscale,
            resultcolor,
            thickness,
            lineType)
        
    return frame

def predict(model, sample):
    with torch.no_grad():
        sample = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)  # add batch dimension
        output = model(sample)
        predicted_class = torch.argmax(output, dim=1).item()
        translated_prediction = translation_dict[predicted_class]

        return translated_prediction


#move out all write_text_on_image

def predict_letter(frame, result):
    "takes the current hand object extruded from the frame and draws the model prediction on the screen"
    try:
        if result.hand_world_landmarks == []: #if 
            frame = write_text_on_image(frame) #only draw saved part
            return frame, None
        else:
            #transform the landmarks to the same format as the classifier is used to
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_world_landmarks[0]])
            landmarks -= landmarks[0]
            scale = np.linalg.norm(landmarks[9] - landmarks[0])  # scales mean bone length
            landmarks /= scale

            landmarks = landmarks.flatten() # is this right?
            prediction = predict(model, landmarks) #we should probably just return the prediction

            frame = write_text_on_image(frame,prediction) #draw the predicted letter on the image
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

def main():
    cap = cv2.VideoCapture(0)

    #get text
    global correct_string 
    global colorvector
    global adjusted_placements
    
    correct_string = get_text().upper() # gets the text to be written from the database

    colorvector = [0] * len(correct_string)

    for pos in range(1,len(correct_string)):
        if correct_string[pos- 1] in skinny_set:
            adjusted_placements.append(15)
        else:
            adjusted_placements.append(40)


    #create face landmarker
    face_landmarker = FaceLandmarkerWrapper()

    # create hand landmarker
    hand_landmarker = Landmarker()

    while True:
        # pull frame
        ret, frame = cap.read()
        # mirror frame
        frame = cv2.flip(frame, 1)
        # update landmarker results
        hand_result = hand_landmarker.detect(frame)
        # draw landmarks on frame
        
        frame, prediction = predict_letter(frame, hand_result) # predicts the letter based on the hand posture and shows the letter on screen
        
        face_landmarks = face_landmarker.detect(frame)

    
        if face_landmarks is not None: #here is where we should check for input?
            frame = face_landmarker.draw(frame, face_landmarks)

            #Should we put input flipflop here?
            inputflag = face_landmarker.take_input(face_landmarks) # returns true if we should enter the character
            if inputflag:
                get_input(prediction)
                write_text_on_image(frame, prediction)


        frame = draw_landmarks_on_image(frame, hand_result) #should perhaps reorder this to draw afterwards

        # display image
        cv2.imshow('frame',frame)

        if cv2.waitKey(1) == ord('q'):
            break

    hand_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
