from pathlib import Path
TRAIN_PATH = None
TEST_PATH = None
OUTPUT_DIR = "results/training"
MODEL_PATH = "models/final_model.h5"
THRESHOLD_PATH = "results/best_threshold.json"
TEST_OUTPUT_DIR  = "results/evaluation"
FGSM_EPS = 0.05
FGSM_FRAC = 0.3
BATCH_SIZE_ADV = 2000
IMG_SIZE = (32, 32)