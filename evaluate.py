import os
import json
import time
import numpy as np
from tensorflow.keras.models import load_model
from config import config
from utils import prepare_dataset, generate_fgsm_subset, print_classification_report, plot_confusion_matrix, plot_roc, download_dataset

# Dataset Downloading
train_path, test_path = download_dataset()
config.TRAIN_PATH = train_path
config.TEST_PATH = test_path
print("Train path:", config.TRAIN_PATH)
print("Test path:", config.TEST_PATH)

os.makedirs(config.TEST_OUTPUT_DIR, exist_ok=True)
with open(config.THRESHOLD_PATH) as f:
    best_threshold = float(json.load(f)["best_threshold"])

# Loading the model for evaluation
model = load_model(config.MODEL_PATH, compile=False)
model.summary()

X_test, y_test = prepare_dataset(config.TEST_PATH)

X_fgsm, y_fgsm = generate_fgsm_subset(model, X_test, y_test)
print(f"Adversarial subset size: {X_fgsm.shape[0]} samples")

def evaluate_and_plot(X, y, phase_name, batch_size=500):
    phase_dir = os.path.join(config.TEST_OUTPUT_DIR, phase_name)
    os.makedirs(phase_dir, exist_ok=True)

    n = len(X)
    all_probs = []
    all_preds = []
    latencies = []

    print(f"\n=== Phase: {phase_name} ({n} samples) ===")

    # Batch-wise predict + latency measurement
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = X[start:end]

        t0 = time.time()
        probs_batch = model.predict(batch, verbose=0).flatten()
        t1 = time.time()

        # record per-sample latency (ms)
        elapsed = (t1 - t0) * 1000
        per_sample = elapsed / len(batch)
        latencies.extend([per_sample] * len(batch))

        all_probs.extend(probs_batch)
        all_preds.extend((probs_batch >= best_threshold).astype(int))

        print(f"  • Processed {start+1}-{end} | batch time {elapsed:.1f}ms | per-sample {per_sample:.2f}ms")

    # Convert to arrays
    probs = np.array(all_probs, dtype=np.float32)
    preds = np.array(all_preds, dtype=int)
    lat_arr = np.array(latencies)

    # Classification metrics
    print_classification_report(y, preds)

    # Latency stats
    avg_latency = lat_arr.mean()
    med_latency = np.median(lat_arr)
    p95_latency = np.percentile(lat_arr, 95)
    print(f"→ Latency (ms) avg={avg_latency:.2f}, med={med_latency:.2f}, 95th={p95_latency:.2f}")

    # Confusion Matrix
    plot_confusion_matrix(y, preds, os.path.join(phase_dir, "confusion_matrix.png"))

    # ROC Curve
    plot_roc(y, probs, os.path.join(phase_dir, "roc_curve.png"))

# Phase 1: Original
evaluate_and_plot(X_test, y_test, "clean")

# Phase 2: FGSM adversarial only
evaluate_and_plot(X_fgsm, y_fgsm, "adversarial")

# Phase 3: Combined
X_comb = np.concatenate([X_test, X_fgsm], axis=0)
y_comb = np.concatenate([y_test, y_fgsm], axis=0)
print(f"Combined test dataset size: {X_comb.shape[0]} samples")
evaluate_and_plot(X_comb, y_comb, "combined")

print("\n✅ Testing complete. Plots saved.")