import numpy as np
import csv
import joblib


AUDIO_FILENAME = "/test/S01_U04.CH4.wav"
FRAME_HOP_SECONDS = 0.01
FEATURES_FILE = "test_features.npy"
MODEL_PATH = "least_squares_model.pkl"
SCALER_PATH = "scaler.pkl"
OUTPUT_CSV = "segments_output_least_squares.csv"

def load_data(features_path, scaler_path):
    X = np.load(features_path)
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)
    return X_scaled

def load_model(model_path):
    w = joblib.load(model_path)
    return w

def predict(X, w):
    X_aug = np.hstack([X, np.ones((X.shape[0], 1))])  
    y_pred = X_aug @ w
    return (y_pred >= 0.5).astype(int)

def extract_segments(predictions, audio_filename, frame_hop):
    segments = []
    current_class = predictions[0]
    start_frame = 0

    for i in range(1, len(predictions)):
        if predictions[i] != current_class:
            start_time = start_frame * frame_hop
            end_time = i * frame_hop
            label = "foreground" if current_class == 1 else "background"
            segments.append([audio_filename, start_time, end_time, label])
            start_frame = i
            current_class = predictions[i]

    # Τελευταίο segment
    end_time = len(predictions) * frame_hop
    label = "foreground" if current_class == 1 else "background"
    segments.append([audio_filename, start_frame * frame_hop, end_time, label])
    return segments

def save_segments(segments, csv_path):
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Audiofile", "start", "end", "class"])
        writer.writerows(segments)
    print(f"✅ Saved segmented output to {csv_path}")


if __name__ == "__main__":
    X = load_data(FEATURES_FILE, SCALER_PATH)
    w = load_model(MODEL_PATH)
    y_pred = predict(X, w)
    segments = extract_segments(y_pred, AUDIO_FILENAME, FRAME_HOP_SECONDS)
    save_segments(segments, OUTPUT_CSV)
