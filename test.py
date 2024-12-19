
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve

def evaluate_model(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(probabilities.cpu().numpy())

    # Generate metrics
    report = classification_report(y_true, y_pred, digits=4)
    matrix = confusion_matrix(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    print("Classification Report:")
    print(report)
    print("
Confusion Matrix:")
    print(matrix)
    print(f"
ROC-AUC Score: {roc_auc:.4f}")

    return report, matrix, roc_auc, precision, recall
