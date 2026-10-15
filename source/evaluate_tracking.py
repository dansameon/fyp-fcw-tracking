import os
import pandas as pd
import numpy as np
import motmetrics as mm

def parse_kitti_gt(label_path, max_frames=50):
    columns = [
        "frame", "track_id", "type", "truncated", "occluded", "alpha",
        "bbox_left", "bbox_top", "bbox_right", "bbox_bottom",
        "height", "width", "length", "loc_x", "loc_y", "loc_z", "rot_y"
    ]
    df = pd.read_csv(label_path, sep=" ", names=columns)
    
    # Restrict to evaluated frame window
    df = df[df["frame"] < max_frames].copy()
    
    # Valid collision obstacles in KITTI
    valid_classes = ["Car", "Van", "Truck", "Pedestrian", "Cyclist"]
    
    gt_df = df[df["type"].isin(valid_classes)].copy()
    gt_df["bb_width"] = gt_df["bbox_right"] - gt_df["bbox_left"]
    gt_df["bb_height"] = gt_df["bbox_bottom"] - gt_df["bbox_top"]
    
    # Store DontCare boxes to prevent penalizing valid background detections
    dontcare_df = df[df["type"] == "DontCare"].copy()
    dontcare_df["bb_width"] = dontcare_df["bbox_right"] - dontcare_df["bbox_left"]
    dontcare_df["bb_height"] = dontcare_df["bbox_bottom"] - dontcare_df["bbox_top"]
    
    return (
        gt_df[["frame", "track_id", "bbox_left", "bbox_top", "bb_width", "bb_height"]],
        dontcare_df[["frame", "bbox_left", "bbox_top", "bb_width", "bb_height"]]
    )

# 1. Load Ground Truth and DontCare
gt_df, dc_df = parse_kitti_gt("data/kitti/labels/training/label_02/0000.txt", max_frames=50)
gt_df = gt_df.set_index(["frame", "track_id"])

# 2. Load ByteTrack Predictions
pred_cols = ["frame", "track_id", "bbox_left", "bbox_top", "bb_width", "bb_height", "conf", "x", "y", "z"]
pred_df = pd.read_csv("outputs/pred_0000.txt", names=pred_cols)
pred_df = pred_df.set_index(["frame", "track_id"])

acc = mm.MOTAccumulator(auto_id=True)

# 3. Frame-by-frame matching
for frame in range(50):
    gt_frame = gt_df.loc[frame] if frame in gt_df.index.get_level_values(0) else pd.DataFrame()
    pred_frame = pred_df.loc[frame] if frame in pred_df.index.get_level_values(0) else pd.DataFrame()
    
    gt_ids = gt_frame.index.tolist() if not gt_frame.empty else []
    pred_ids = pred_frame.index.tolist() if not pred_frame.empty else []
    
    gt_boxes = gt_frame[["bbox_left", "bbox_top", "bb_width", "bb_height"]].values if not gt_frame.empty else np.empty((0, 4))
    pred_boxes = pred_frame[["bbox_left", "bbox_top", "bb_width", "bb_height"]].values if not pred_frame.empty else np.empty((0, 4))
    
    # Filter out predictions that fall inside DontCare areas before scoring
    dc_frame = dc_df[dc_df["frame"] == frame]
    if not dc_frame.empty and len(pred_boxes) > 0:
        dc_boxes = dc_frame[["bbox_left", "bbox_top", "bb_width", "bb_height"]].values
        # Compute overlap with DontCare regions
        dc_ious = mm.distances.iou_matrix(dc_boxes, pred_boxes, max_iou=0.5)
        # Any prediction with IoU > 0.3 with a DontCare region is ignored, not penalized
        valid_pred_mask = np.ones(len(pred_boxes), dtype=bool)
        for p_idx in range(len(pred_boxes)):
            if np.any(dc_ious[:, p_idx] < 0.7):  # Distance < 0.7 means IoU > 0.3
                valid_pred_mask[p_idx] = False
        
        pred_boxes = pred_boxes[valid_pred_mask]
        pred_ids = [pred_ids[i] for i, valid in enumerate(valid_pred_mask) if valid]

    # Calculate distance matrix (IoU threshold 0.5)
    distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
    acc.update(gt_ids, pred_ids, distances)

mh = mm.metrics.create()
summary = mh.compute(
    acc,
    metrics=["mota", "motp", "idf1", "num_false_positives", "num_misses", "num_switches"],
    name="Seq_0000"
)

summary["motp"] = (1 - summary["motp"]) * 100
summary["mota"] = summary["mota"] * 100
summary["idf1"] = summary["idf1"] * 100

print("\n--- Calibrated Baseline Tracking Results ---")
print(summary.to_string(formatters={
    "mota": "{:.1f}%".format, "motp": "{:.1f}%".format, "idf1": "{:.1f}%".format
}))