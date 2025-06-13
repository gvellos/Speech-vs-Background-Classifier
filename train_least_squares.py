import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def train_least_squares(X, y):
    X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
    w = np.linalg.pinv(X_aug.T @ X_aug) @ X_aug.T @ y
    return w

def predict(X, w):
    X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
    y_pred = X_aug @ w
    return (y_pred >= 0.5).astype(int)

if __name__ == "__main__":
    # Φόρτωση χαρακτηριστικών από .npz
    data = np.load('training_data.npz')
    X = data['X']
    y = data['y']

    # Εκτύπωση αρχικής κατανομής
    unique, counts = np.unique(y, return_counts=True)
    print("Original class distribution:", dict(zip(unique, counts)))

    # Διαχωρισμός των δύο κλάσεων
    X_bg = X[y == 0]
    X_fg = X[y == 1]

    min_len = min(len(X_bg), len(X_fg))

    # Undersample της πλειοψηφούσας κλάσης
    X_bg_down = resample(X_bg, replace=False, n_samples=min_len, random_state=42)
    X_fg_down = resample(X_fg, replace=False, n_samples=min_len, random_state=42)

    # Συνένωση και δημιουργία y
    X_balanced = np.vstack([X_bg_down, X_fg_down])
    y_balanced = np.hstack([
        np.zeros(min_len, dtype=int),
        np.ones(min_len, dtype=int)
    ])

    print("Balanced class distribution:", dict(zip(*np.unique(y_balanced, return_counts=True))))

    # Κανονικοποίηση
    scaler = StandardScaler()
    X_balanced = scaler.fit_transform(X_balanced)

    # Διαχωρισμός σε train/test με stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.3, stratify=y_balanced, random_state=42
    )

    # Εκπαίδευση
    w = train_least_squares(X_train, y_train)

    # Πρόβλεψη
    y_pred = predict(X_test, w)

    # Αξιολόγηση
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nAccuracy: {acc * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)

    # Αναλυτικά metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["background", "foreground"]))

    # Οπτικοποίηση confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["background", "foreground"],
                yticklabels=["background", "foreground"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # Αποθήκευση μοντέλου (w) και scaler
    joblib.dump(w, "least_squares_model.pkl")
    joblib.dump(scaler, "scaler_least_squares.pkl")

    print("✅ Model and scaler saved.")
