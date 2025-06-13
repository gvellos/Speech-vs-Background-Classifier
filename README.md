# 🔊 Speech vs Background Classifier

This project classifies audio frames into **speech (foreground)** or **background (noise)** using extracted audio features and a **Least Squares Classifier**.

---

## 📁 Project Structure

```
.
├── data/
│   ├── train/
│   │   ├── speech/
│   │   └── noise/
├── features/
│   └── test_features.npy
├── models/
│   ├── scaler.pkl
│   └── least_squares_model.pkl
├── outputs/
│   └── segments_output_least_squares.csv
├── training_data.npz
├── train_least_squares.py
├── extract_features.py
├── least_squares_predict.py
└── README.md
```

---

## 🔧 1. Feature Extraction

```bash
python extract_features.py
```

Extracts features using `librosa`:

* 13 MFCCs
* Zero Crossing Rate
* RMS Energy
* Spectral Centroid

➡️ Saves to `features/test_features.npy`

---

## 🧠 2. Train Least Squares Classifier

```bash
python train_least_squares.py
```

Workflow:

* Load `training_data.npz`
* Downsample majority class for balance
* Standardize features
* Train least squares weights `w`
* Evaluate (Accuracy, F1, Confusion Matrix)

➡️ Saves:

* `models/scaler.pkl`
* `models/least_squares_model.pkl`

---

## 🔍 3. Predict & Extract Segments

```bash
python least_squares_predict.py
```

Workflow:

* Load and scale test features
* Load trained model weights
* Predict class per frame
* Merge consecutive frames into segments
* Save to CSV

➡️ Example output:

```csv
Audiofile,start,end,class
S01_U04.CH4.wav,0.00,1.24,background
S01_U04.CH4.wav,1.24,2.01,foreground
...
```

---

## 🛠 Requirements

* Python 3.8+
* `numpy`, `librosa`, `scikit-learn`, `joblib`, `matplotlib`, `seaborn`

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📌 Notes

* Frame hop duration (e.g. 10ms) must remain consistent across stages
* Least Squares is a simple linear baseline — use an MLP for better results (see `mlp_model.pth` version)

---

## 👤 Author

**George Vellos**

University of Piraeus
