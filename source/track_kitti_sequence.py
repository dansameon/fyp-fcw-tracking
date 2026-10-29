import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# --- TTC Memory Variables ---
track_history = {}  
FPS = 30 
dt = 1.0 / FPS

# --- 1. DEFINING THE FUNCTIONS ---

def is_in_ego_lane(x_center, y_bottom, frame_width, frame_height):
    # Pushed the top edge up to 40% of the screen height and widened the base
    roi_corners = np.array([[(int(frame_width * 0.15), frame_height), 
                             (int(frame_width * 0.40), int(frame_height * 0.40)), 
                             (int(frame_width * 0.60), int(frame_height * 0.40)), 
                             (int(frame_width * 0.85), frame_height)]], dtype=np.int32)
    
    # Check if the bottom-center of the vehicle (the tires) is inside the polygon
    # cv2.pointPolygonTest returns >= 0 if the point is inside the shape
    result = cv2.pointPolygonTest(roi_corners, (int(x_center), int(y_bottom)), False)
    return result >= 0, roi_corners

def calculate_ttc(track_id, current_width):
    if track_id not in track_history:
        track_history[track_id] = {'prev_width': current_width, 'smooth_rate': 0}
        return None, "Calc..."

    prev_width = track_history[track_id]['prev_width']
    raw_expansion_rate = (current_width - prev_width) / dt
    
    smooth_rate = (0.7 * track_history[track_id]['smooth_rate']) + (0.3 * raw_expansion_rate)
    
    track_history[track_id]['prev_width'] = current_width
    track_history[track_id]['smooth_rate'] = smooth_rate

    # Guard: Receding or static targets
    if smooth_rate <= 1.0: 
        return None, "Safe"

    ttc = current_width / smooth_rate
    return ttc, f"{ttc:.2f}s"

# --- 2. PIPELINE INITIALIZATION ---

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

VEHICLE_CLASSES = [0, 1, 2, 3, 5, 7]

output_dir = "outputs/sequence_0000_tracked"
os.makedirs(output_dir, exist_ok=True)
pred_file = "outputs/pred_0000.txt"

# --- 3. MAIN EXECUTION LOOP ---

with open(pred_file, "w") as f:
    for idx, frame_path in enumerate(frame_files[:50]):
        
        # Read the raw image
        frame = cv2.imread(frame_path)
        frame_h, frame_w = frame.shape[:2]

        # Run ByteTrack on the ORIGINAL, unmasked frame
        results = model.track(
            source=frame, persist=True, tracker="bytetrack.yaml",
            device=device, classes=VEHICLE_CLASSES, conf=0.45, iou=0.40, verbose=False
        )
        
        # Create a copy of the frame to draw our own custom boxes and ROI on
        annotated_frame = frame.copy()
        
        # Draw the blue ROI polygon on the screen so you can verify the lane placement
        _, roi_pts = is_in_ego_lane(0, 0, frame_w, frame_h)
        cv2.polylines(annotated_frame, [roi_pts], isClosed=True, color=(255, 0, 0), thickness=2)

        boxes = results[0].boxes
        if boxes.id is not None:
            for box, track_id in zip(boxes.xywh, boxes.id):
                x_c, y_c, w, h = box.tolist()
                
                # Find the bottom center of the bounding box (where the tires touch the road)
                y_bottom = y_c + (h / 2)
                
                # Filter: Only process if the tires are inside our Ego-Lane ROI
                in_lane, _ = is_in_ego_lane(x_c, y_bottom, frame_w, frame_h)
                
                if in_lane:
                    x_left = x_c - (w / 2)
                    y_top = y_c - (h / 2)
                    
                    # Calculate TTC
                    ttc_value, ttc_text = calculate_ttc(int(track_id), w)
                    
                    # Draw green bounding box and red TTC text
                    cv2.rectangle(annotated_frame, (int(x_left), int(y_top)), (int(x_left+w), int(y_top+h)), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"ID:{int(track_id)} | TTC:{ttc_text}", (int(x_left), int(y_top) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Write only valid ego-lane vehicles to the evaluation file
                    f.write(f"{idx},{int(track_id)},{x_left:.2f},{y_top:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")

        # Save the final annotated image
        cv2.imwrite(os.path.join(output_dir, f"tracked_{idx:06d}.png"), annotated_frame)

print(f"\nTracking complete! Predictions saved to {pred_file}")