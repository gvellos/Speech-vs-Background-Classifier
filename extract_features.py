import os
import librosa
import numpy as np
from tqdm import tqdm

FRAME_SIZE = 0.030  # 30ms
HOP_SIZE = 0.010    # 10ms
SR = 16000          # Sample Rate
N_MFCC = 13

def extract_features_from_file(file_path, label, sr=SR, n_mfcc=N_MFCC):
    y, _ = librosa.load(file_path, sr=sr)
    frame_length = int(FRAME_SIZE * sr)
    hop_length = int(HOP_SIZE * sr)

    if y.shape[0] < frame_length:
        return np.empty((0, n_mfcc*3 + 3)), np.empty((0,))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                               hop_length=hop_length, n_fft=frame_length)
    if mfcc.shape[1] < 9:
        return np.empty((0, n_mfcc*3 + 3)), np.empty((0,))

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length, frame_length=frame_length)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)

    features = np.vstack([mfcc, delta, delta2, zcr, spec_centroid, spec_bw]).T
    labels = np.full((features.shape[0],), label, dtype=np.uint8)

    return features, labels

def load_dataset(speech_dir, noise_dir):
    X = []
    y = []

    for label_dir, label in [(speech_dir, 1), (noise_dir, 0)]:
        wav_files = []
        for root, _, files in os.walk(label_dir):
            for file in files:
                if file.endswith('.wav'):
                    wav_files.append(os.path.join(root, file))

        wav_files.sort()
        print(f"Loading from {label_dir} ({len(wav_files)} files)...")

        for fpath in tqdm(wav_files):
            feats, labels = extract_features_from_file(fpath, label)
            if feats.shape[0] == 0:
                continue
            X.append(feats)
            y.append(labels)

    X = np.vstack(X)
    y = np.concatenate(y)
    print(f"Final dataset shape: {X.shape}, labels: {y.shape}")
    return X, y

if __name__ == "__main__":
    speech_path = "/train/speech"
    noise_path = "/train/noise"
    X, y = load_dataset(speech_path, noise_path)
    np.savez("training_data.npz", X=X, y=y)
