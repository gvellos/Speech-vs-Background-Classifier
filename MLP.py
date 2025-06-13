import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Ορισμός του MLP μοντέλου
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  
        )

    def forward(self, x):
        return self.model(x)

# Εκπαίδευση
def train(model, criterion, optimizer, X_train, y_train, X_test, y_test, epochs=10):
    for epoch in range(epochs):
        model.train()
        outputs = model(X_train).view(-1)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test).view(-1)
            probs = torch.sigmoid(test_outputs)
            preds = (probs >= 0.4).float()
            acc = accuracy_score(y_test.cpu(), preds.cpu())
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f} - Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":



    # Φόρτωση δεδομένων
    data = np.load("training_data.npz")
    X = data["X"]
    y = data["y"]

    # Standardization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Τανσορποίηση
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Δημιουργία μοντέλου
    model = MLP(input_dim=X.shape[1])

    # Υπολογισμός pos_weight για να εξισορροπηθεί η κλάση
    num_pos = y_train.sum()
    num_neg = len(y_train) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Εκπαίδευση
    train(model, criterion, optimizer, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, epochs=10)

    # Τελική αξιολόγηση
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor).view(-1)
        probs = torch.sigmoid(logits)
        y_pred_class = (probs >= 0.5).int().cpu().numpy()

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_class, target_names=["background", "foreground"]))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_class)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["background", "foreground"],
                yticklabels=["background", "foreground"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()



    # Αποθήκευση μοντέλου και scaler
    torch.save(model.state_dict(), "mlp_model.pth")
    print("✅ Το μοντέλο αποθηκεύτηκε ως mlp_model.pth")

    joblib.dump(scaler, "scaler.pkl")
    print("✅ Ο scaler αποθηκεύτηκε ως scaler.pkl")