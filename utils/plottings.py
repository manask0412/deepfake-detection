import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score, precision_recall_curve
)

def print_classification_report(y_true, y_pred):
    rpt = classification_report(y_true, y_pred, digits=4)
    mismatches = np.sum(y_pred != y_true)
    total = len(y_true)
    print(rpt)
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_true, y_pred):.4f}")
    print(f"Mismatches: {mismatches}/{total} ({mismatches/total*100:.2f}%)")

def plot_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    # Create the heatmap plot using seaborn
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=["Real", "Fake"], 
                yticklabels=["Real", "Fake"], 
                cbar=True, cmap="Blues")
    # Title and labels
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    # Save the plot
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    plt.close()

def plot_roc(y_true, y_probs, path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0,1],[0,1],"--")
    plt.title("ROC Curve")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    plt.close()

def plot_loss_acc(history, path):
    plt.figure(figsize=(12, 6))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    plt.close()

def find_best_threshold(y_true, y_probs, path_curve):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1 = 2*(precision*recall)/(precision+recall+1e-8)
    idx = np.nanargmax(f1)
    best_t = thresholds[idx]
    plt.figure(figsize=(4,4))
    plt.plot(recall, precision, marker='.')
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path_curve)
    plt.show()
    plt.close()
    return best_t