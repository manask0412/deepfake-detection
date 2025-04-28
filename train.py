import os
import numpy as np
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from config import config
from utils import prepare_dataset, augment_dataset, generate_adversarial_examples, find_best_threshold, download_dataset
from trainer import build_model, compile_and_train
# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Dataset Downloading
train_path, test_path = download_dataset()
config.TRAIN_PATH = train_path
config.TEST_PATH = test_path
print("Train path:", config.TRAIN_PATH)
print("Test path:", config.TEST_PATH)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# Load & split
X, y = prepare_dataset(config.TRAIN_PATH)
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=42)

print(f"Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}")
print(f"Train REAL/FAKE: {np.sum(y_train==0)}/{np.sum(y_train==1)}")
print(f"Val   REAL/FAKE: {np.sum(y_val==0)}/{np.sum(y_val==1)}")

# Augment clean
X_train_aug, y_train_aug = augment_dataset(X_train, y_train, n_rounds=1)

# Build model
model, base_model = build_model()
model.summary()

# Phase 1: Initial clean training
model, hist1 = compile_and_train(
    model, base_model,
    X_train_aug, y_train_aug, X_val, y_val,
    phase_name="initial", epochs=20,
    fine_tune=False, threshold=0.5)

# Phase 2: Adversarial training (FGSM only)
X_adv_all, y_adv_all = generate_adversarial_examples(
    model, X_train, y_train,
    attack_type="FGSM", eps=0.05, sample_frac=0.3, batch_size=2000)

# Combine with augmented clean
X_adv_comb = np.concatenate([X_train_aug, X_adv_all[len(X_train):]], axis=0)
y_adv_comb = np.concatenate([y_train_aug, y_adv_all[len(y_train):]], axis=0)
print(f"Combined adv training set size: {X_adv_comb.shape[0]} samples")

model, hist2 = compile_and_train(
    model, base_model,
    X_adv_comb, y_adv_comb, X_val, y_val,
    phase_name="adv_training", epochs=5,
    fine_tune=False, threshold=0.5)

# Phase 3: Fine-tuning on clean again
model, hist3 = compile_and_train(
    model, base_model,
    X_train_aug, y_train_aug, X_val, y_val,
    phase_name="finetune", epochs=10,
    fine_tune=True, threshold=0.5)

os.makedirs("models", exist_ok=True)
model.save("models/final_model.h5")
print("Model saved after Final Training Phase")

# Phase 4: Final threshold tuning & save
y_val_probs = model.predict(X_val).flatten()
best_thr = find_best_threshold(
    y_val, y_val_probs,
    os.path.join("results", "precision_recall_curve_final.png"))

THRESHOLD_FILE  = os.path.join("results", "best_threshold.json")
with open(THRESHOLD_FILE, "w") as f:
    json.dump({"best_threshold": float(best_thr)}, f)
print(f"Best threshold saved: {best_thr}")

print("Done. Final model and threshold saved.")