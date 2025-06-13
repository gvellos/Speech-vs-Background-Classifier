# 🔊 Speech vs Background Segmentation System

This project implements an audio segmentation system that detects and separates **speech** (foreground) from **background noise** in a mixed audio signal. The system performs frame-wise classification using two models — **Least Squares** and a **3-layer MLP** — followed by post-processing to produce continuous segments. Evaluation is conducted using ground-truth annotations to measure accuracy, precision, recall, and F1-score.

---

## 📁 Project Structure

```
├── data/
│   ├── train/
│   │   ├── speech/
│   │   └── noise/
│   ├── test/
│   │   └── S01_U04.CH4.wav
│   └── transcriptions/
│       └── S01.json
├── extract_features.py
├── extract_WAVfeatures.py
├── least_squares.py
├── MLP.py
├── post_processingLS.py
├── post_processingMLP.py
├── evaluate_segments.py
├── requirements.txt
└── README.md
```

## 🚀 Steps to Run

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Extract features** from audio:

   ```bash
   python extract_features.py
   python extract_WAVfeatures.py
   ```

3. **Train the models**:

   ```bash
   python least_squares.py
   python MLP.py
   ```

4. **Run post-processing** using trained models:

   ```bash
   python post_processingLS.py
   python post_processingMLP.py
   ```

5. **Evaluate segmentation performance**:

   ```bash
   python evaluate_segments.py
   ```

## Output

* Segmentation results are saved in `.csv` format with columns:

  ```
  Audiofile, start, end, class
  ```
* Evaluation includes frame-level metrics: **Accuracy**, **Precision**, **Recall**, and **F1-score**.

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
