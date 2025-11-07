from torch.utils.data import DataLoader, random_split
from dataset_class import handLandmarkDataset
import torch
import torch.nn as nn
from model import ASL_MLP
import logging

# Load full dataset
dataset = handLandmarkDataset("training_landmarks")

# ---- Create Train/Validation Split ----
train_size = int(0.8 * len(dataset))   # 80% for training
val_size = len(dataset) - train_size   # 20% for validation
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
# -------------- Logging ----------------------------
logging.basicConfig(
    filename="training_log.txt",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)
# -------------------------------------------

model = ASL_MLP()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)   # Move model to device  ← IMPORTANT

for epoch in range(25):
    # ---------- TRAINING ----------
    model.train()
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    # ---------- VALIDATION ----------
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            val_loss += criterion(out, y).item()

            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    val_loss /= len(val_loader)
    val_accuracy = correct / total

    printinfo = f"Epoch {epoch+1} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2%}"

    logger.info(printinfo)
    print(printinfo)

# Save model
torch.save(model.state_dict(), "asl_mlp_model25.pth")
print("Model saved as asl_mlp_model25.pth")
