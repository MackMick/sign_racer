import cv2
import time
import mediapipe as mp
from landmarker_class import Landmarker
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

from model import ASL_MLP
import torch

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 4
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

correct_string = "THIS IS A SAMPLE"
current_pos = 0
colorvector = [0] * len(correct_string) # makes empty array -> 
#0 -> not yet seen, 1 -> correct, 2 -> wrong
#we should really make this a real lookup thing instead of just arbitrarily ascribing colors to numbers

fullstring = ""

skinny_set = {"I"}

adjusted_placements = [20] # list to make sure all letters are placed in a good manner -> distinguish between thin and wide letters
for pos in range(1,len(correct_string)):
    if correct_string[pos- 1] in skinny_set:
        adjusted_placements.append(15)
    else:
        adjusted_placements.append(40)


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

    if len(text) > 0:
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
        all_ratings = []
        sample = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)  # add batch dimension
        output = model(sample)
        predicted_class = torch.argmax(output, dim=1).item()

        for idx, val in enumerate(output[0]):
            all_ratings.append((translation_dict[idx], round(float(val), 3)))


        
        translated_prediction = translation_dict[predicted_class]
        
        return translated_prediction, all_ratings

def predict_letter(frame, result):
    "takes the current hand object extruded from the frame and draws the model prediction on the screen"
    try:
        if result.hand_world_landmarks == []:
            frame = write_text_on_image(frame) #only draw saved part
            return frame
        else:
            #transform the landmarks to the same format as the classifier is used to
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_world_landmarks[0]])
            landmarks -= landmarks[0]
            scale = np.linalg.norm(landmarks[9] - landmarks[0])  # scales mean bone length
            landmarks /= scale
            global fullstring
            global correct_string
            global colorvector

            landmarks = landmarks.flatten() # is this right?
            prediction, all_ratings = predict(model, landmarks)


            if cv2.waitKey(1) == ord('a') and len(fullstring) <= len(correct_string): #change this to blinking somehow
                fullstring += prediction

                if prediction == correct_string[len(fullstring)-1]:
                    colorvector[len(fullstring)-1] = 1
                else:
                    colorvector[len(fullstring)-1] = 2
                print(all_ratings)



            frame = write_text_on_image(frame,prediction) #draw the predicted letter on the image
        return frame #add the things to the frame properly
    except Exception as e:
        print("Predict error: ", e)
        return frame


def main():
    cap = cv2.VideoCapture(0)

    # create landmarker
    hand_landmarker = Landmarker()

    while True:
        # pull frame
        ret, frame = cap.read()
        # mirror frame
        frame = cv2.flip(frame, 1)
        # update landmarker results
        hand_landmarker.detect_async(frame)
        # draw landmarks on frame
        
        frame = predict_letter(frame, hand_landmarker.result) # predicts the letter based on the hand posture and shows the letter on screen
        
        frame = draw_landmarks_on_image(frame, hand_landmarker.result) #should perhaps reorder this to draw afterwards

        # display image
        cv2.imshow('frame',frame)

        if cv2.waitKey(1) == ord('q'):
            break

    # release everything
    hand_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
