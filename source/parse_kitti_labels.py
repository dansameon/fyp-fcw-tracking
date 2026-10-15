import os
import pandas as pd

def parse_kitti_tracking_label(label_path):
    """
    Parses KITTI tracking label .txt into standard 2D tracking columns.
    KITTI columns: frame, track_id, type, truncated, occluded, alpha,
                   bbox_left, bbox_top, bbox_right, bbox_bottom, height, width, length, x, y, z, rot_y
    """
    columns = [
        "frame", "track_id", "type", "truncated", "occluded", "alpha",
        "bbox_left", "bbox_top", "bbox_right", "bbox_bottom",
        "height", "width", "length", "loc_x", "loc_y", "loc_z", "rot_y"
    ]
    
    df = pd.read_csv(label_path, sep=" ", names=columns)
    
    # Obstacle categories evaluated in driving benchmarks
    target_classes = ["Car", "Van", "Truck", "Pedestrian", "Cyclist"]
    filtered_df = df[df["type"].isin(target_classes)].copy()
    
    # Convert [xmin, ymin, xmax, ymax] into MOT challenge format [x, y, width, height]
    filtered_df["bb_width"] = filtered_df["bbox_right"] - filtered_df["bbox_left"]
    filtered_df["bb_height"] = filtered_df["bbox_bottom"] - filtered_df["bbox_top"]
    
    return filtered_df[["frame", "track_id", "type", "bbox_left", "bbox_top", "bb_width", "bb_height"]]

if __name__ == "__main__":
    label_file = "data/kitti/labels/training/label_02/0000.txt"
    
    if os.path.exists(label_file):
        gt = parse_kitti_tracking_label(label_file)
        print("KITTI Ground-Truth Parsed Successfully:")
        print(f"Total labeled objects across sequence 0000: {len(gt)}")
        print("\nFirst 10 Ground-Truth Bounding Boxes:")
        print(gt.head(10))
    else:
        print(f"File not found: {label_file}")