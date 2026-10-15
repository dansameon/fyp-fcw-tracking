import os
import cv2
import torch
from ultralytics import YOLO

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Tracking using device: {device}")

model = YOLO("yolov8n.pt")

sequence_dir = "data/kitti/images/training/image_02/0000"
if not os.path.exists(sequence_dir):
    raise FileNotFoundError(f"Sequence directory not found: {sequence_dir}")

frame_files = sorted(
    [os.path.join(sequence_dir, f) for f in os.listdir(sequence_dir) if f.endswith(".png")]
)
print(f"Found {len(frame_files)} frames in sequence 0000.")

# Target obstacle classes: person, bicycle, car, motorcycle, bus, truck
VEHICLE_CLASSES = [0, 1, 2, 3, 5, 7]

output_dir = "outputs/sequence_0000_tracked"
os.makedirs(output_dir, exist_ok=True)

# NEW: Open a text file to save our predictions
pred_file = "outputs/pred_0000.txt"
with open(pred_file, "w") as f:
    for idx, frame_path in enumerate(frame_files[:50]):
        results = model.track(
            source=frame_path, persist=True, tracker="bytetrack.yaml",
            device=device, classes=VEHICLE_CLASSES, conf=0.45, iou=0.5, verbose=False
        )
        
        # Save annotated image
        annotated_frame = results[0].plot()
        cv2.imwrite(os.path.join(output_dir, f"tracked_{idx:06d}.png"), annotated_frame)

        # NEW: Write coordinates to text file
        boxes = results[0].boxes
        if boxes.id is not None:
            for box, track_id in zip(boxes.xywh, boxes.id):
                # Convert YOLO (center_x, center_y) to MOT format (left_x, top_y)
                x_c, y_c, w, h = box.tolist()
                x_left = x_c - (w / 2)
                y_top = y_c - (h / 2)
                
                # Format: frame, track_id, x, y, width, height, confidence, -1, -1, -1
                f.write(f"{idx},{int(track_id)},{x_left:.2f},{y_top:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")

print(f"\nTracking complete! Predictions saved to {pred_file}")