# 🔊 Speech vs Background Segmentation System

This project implements an audio segmentation system that detects and separates **speech** (foreground) from **background noise** in a mixed audio signal. The system performs frame-wise classification using two models — **Least Squares** and a **3-layer MLP** — followed by post-processing to produce continuous segments. Evaluation is conducted using ground-truth annotations to measure accuracy, precision, recall, and F1-score.

---

## 📁 Project Structure

```
project/
│
├── data/
│   ├── train/
│   │   ├── speech/                    # Training speech wav files
│   │   └── noise/                     # Training noise wav files
│   ├── test/
│   │   ├── S01_U04.CH4.wav            # Mixed test audio
│   │   └── transcriptions/
│   │       └── S01.json               # Ground truth annotations
│
├── features/
│   ├── train_features.npy             # Extracted training features
│   └── test_features.npy              # Extracted test features
│
├── models/
│   ├── least_squares_model.pkl       # Saved Least Squares model
│   ├── mlp_model.pkl                  # Saved MLP model
│   └── scaler.pkl                     # Scaler used for feature normalization
│
├── output/
│   ├── segments_output_least_squares.csv  # Segments predicted by Least Squares
│   └── segments_output_mlp.csv            # Segments predicted by MLP
│
├── extract_features.py               # Extracts MFCC, ZCR, RMS, Centroid features
├── train_least_squares.py           # Trains the Least Squares classifier
├── train_mlp.py                      # Trains the 3-layer MLP classifier
├── predict_and_segment.py           # Performs classification and segmentation
├── evaluate.py                       # Evaluates output segments vs ground truth
└── README.md
```

---

## 🚀 How to Run

1. **Extract Features**

   ```bash
   python extract_features.py
   ```

2. **Train Classifiers**

   * Least Squares:

     ```bash
     python train_least_squares.py
     ```
   * MLP:

     ```bash
     python train_mlp.py
     ```

3. **Segment the Test Audio**

   ```bash
   python predict_and_segment.py
   ```

4. **Evaluate Results**

   ```bash
   python evaluate.py
   ```

---

## 📊 Sample Evaluation Metrics

### Least Squares Classifier:

* Accuracy : 0.5562
* Precision: 0.8200
* Recall   : 0.5567
* F1-score : 0.6632

### MLP Classifier:

* Accuracy : 0.7813
* Precision: 0.8554
* Recall   : 0.8681
* F1-score : 0.8617

---

## 📄 Output Format

Each classifier produces a CSV file with the following format:

```
Audiofile, start, end, class
S01_U04.CH4.wav, 0.00, 2.21, background
S01_U04.CH4.wav, 2.21, 5.73, foreground
S01_U04.CH4.wav, 5.73, 8.00, background
...
```

---

## 📚 Datasets Used

* [MUSAN Corpus](https://www.openslr.org/17/)
* [CHiME Challenge Data](https://www.chimechallenge.org/)

---

## 💻 Dependencies

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

---

## 👤 Author

George Vellos
University of Piraeus
