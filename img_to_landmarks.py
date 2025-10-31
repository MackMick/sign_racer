import mediapipe as mp
import numpy as np
import os
import time
import kagglehub

root_path = kagglehub.dataset_download("kapillondhe/american-sign-language")
model_path = "hand_landmarker.task"
print("Path to dataset files:", root_path)

#root_path = root_path + "/ASL_Dataset/Train"
root_path = root_path + "/ASL_Dataset/Test"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the image mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.CPU),
    running_mode=VisionRunningMode.IMAGE)
 
for class_name in sorted(os.listdir(root_path)):
    class_folder = os.path.join(root_path, class_name)

    for filename in os.listdir(class_folder):
        file_source = os.path.join(class_folder, filename)
        time.sleep(0.005)

        with HandLandmarker.create_from_options(options) as landmarker:
            mp_image = mp.Image.create_from_file(file_source)
            hand_landmarker_result = landmarker.detect(mp_image) #detect_async
            result = hand_landmarker_result
    
        if result.hand_world_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_world_landmarks[0]])

            landmarks -= landmarks[0]

            scale = np.linalg.norm(landmarks[9] - landmarks[0])  # scales mean bone length
            landmarks /= scale

            savepath = f"testing_landmarks/{class_name}/{filename.split(".")[0]}.npy"
            os.makedirs(os.path.dirname(savepath), exist_ok=True)

            with open(savepath, "wb") as f:
                np.save(f, landmarks)
