import numpy as np
import librosa

def extract_features_from_signal(y, sr, frame_length, frame_step, n_mfcc):
    fl = int(frame_length * sr)
    hl = int(frame_step * sr)
    n_fft = fl

    if y.shape[0] < fl:
        return np.empty((0, n_mfcc + 3))

    mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                     n_fft=n_fft, hop_length=hl, center=False)
    zcr      = librosa.feature.zero_crossing_rate(y, frame_length=fl, hop_length=hl, center=False)
    rms      = librosa.feature.rms(y=y, frame_length=fl, hop_length=hl, center=False)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr,
                                                  n_fft=n_fft, hop_length=hl, center=False)

    return np.vstack([mfcc, zcr, rms, centroid]).T


wav_path = "/test/S01_U04.CH4.wav"
output_path = "test_features.npy"

sample_rate = 16000
frame_length = 0.03  # 30 ms
frame_step = 0.01    # 10 ms
n_mfcc = 13

y, sr = librosa.load(wav_path, sr=sample_rate)
features = extract_features_from_signal(y, sr, frame_length, frame_step, n_mfcc)

np.save(output_path, features)
print(f"✅ Features saved to '{output_path}' with shape: {features.shape}")
