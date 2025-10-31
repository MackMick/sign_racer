import torch
import torch.nn as nn

class ASL_MLP(nn.Module):
    def __init__(self):
        super(ASL_MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(63, 128),   # Input layer (63 numbers: 21 landmarks * 3 coords)
            nn.ReLU(),
            nn.Linear(128, 64),   # Hidden layer
            nn.ReLU(),
            nn.Linear(64, 27)     # Output layer (26 letters A–Z)
        )

    def forward(self, x):
        return self.model(x)
