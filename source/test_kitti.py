import os
import torch
from ultralytics import YOLO

# 1. Verify GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")

# 2. Load model
model = YOLO("yolov8n.pt")

# 3. Path to the very first frame of Sequence 0000
sample_frame = "data/kitti/images/training/image_02/0000/000000.png"

if not os.path.exists(sample_frame):
    raise FileNotFoundError(f"Frame not found at {sample_frame}. Check directory path.")

# 4. Run inference & save visual output
results = model.predict(
    source=sample_frame,
    device=device,
    conf=0.25,
    save=True,
    project="outputs",
    name="smoke_test"
)

print("\nInference successful! Check 'outputs/smoke_test' for the annotated frame.")