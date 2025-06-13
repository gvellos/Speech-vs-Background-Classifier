import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

FRAME_STEP = 0.01  
JSON_PATH = "test/transcriptions/S01.json"

def time_str_to_sec(t):
    h, m, s = t.split(":")
    return float(h) * 3600 + float(m) * 60 + float(s)

def evaluate_segmentation(csv_path, json_path=JSON_PATH, frame_step=FRAME_STEP):
    # Φόρτωση προβλέψεων από CSV
    pred_df = pd.read_csv(csv_path)
    total_duration = pred_df["end"].max()
    n_frames = int(np.ceil(total_duration / frame_step))
    pred_labels = np.zeros(n_frames, dtype=int)

    for _, row in pred_df.iterrows():
        start_frame = int(np.floor(row["start"] / frame_step))
        end_frame = int(np.ceil(row["end"] / frame_step))
        label = 1 if row["class"] == "foreground" else 0
        pred_labels[start_frame:end_frame] = label

    # Φόρτωση ground truth από το JSON
    with open(json_path, "r") as f:
        gt_json = json.load(f)

    gt_labels = np.zeros(n_frames, dtype=int)
    for seg in gt_json:
        start = time_str_to_sec(seg["start_time"])
        end = time_str_to_sec(seg["end_time"])
        start_frame = int(np.floor(start / frame_step))
        end_frame = int(np.ceil(end / frame_step))
        gt_labels[start_frame:end_frame] = 1


    acc = accuracy_score(gt_labels, pred_labels)
    prec = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)
    f1 = f1_score(gt_labels, pred_labels)

    print(f"\n📊 Frame-level Evaluation Metrics for '{csv_path}':")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

evaluate_segmentation("segments_output_least_squares.csv")
evaluate_segmentation("segments_output_mlp.csv")