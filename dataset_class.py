import os
from torch.utils.data import Dataset
import numpy as np


class handLandmarkDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.labels = sorted(os.listdir(root_dir))
        self.samples = [] # append each npy path and letter in terms of an id as a tuple to the thing
        
        for label_idx, label in enumerate(self.labels):
            class_folder = os.path.join(root_dir, label)
            for file in os.listdir(class_folder):
                npy_path = os.path.join(class_folder, file)
                self.samples.append((npy_path, label_idx))

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        npy_path, label_idx = self.samples[idx]
        x = np.load(npy_path)

        # flatten if needed
        x = x.reshape(-1).astype(np.float32)

        return x, label_idx


test = handLandmarkDataset("training_landmarks/")
