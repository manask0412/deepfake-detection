import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import BinaryCrossentropy
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import FastGradientMethod
from config import config


def load_images_from_folder(folder, label, img_size=(32, 32)):
    imgs, lbls = [], []
    print(f"Loading images from path: {folder}")
    for fname in sorted(os.listdir(folder)):
        img = load_img(os.path.join(folder, fname), target_size=img_size)
        arr = img_to_array(img)
        arr = preprocess_input(arr)
        imgs.append(arr); lbls.append(label)
    return imgs, lbls

def prepare_dataset(base_path):
    real_x, real_y = load_images_from_folder(os.path.join(base_path, "REAL"), 0)
    fake_x, fake_y = load_images_from_folder(os.path.join(base_path, "FAKE"), 1)
    X = np.array(real_x + fake_x, dtype=np.float32)
    y = np.array(real_y + fake_y, dtype=np.int32)
    print(f"  → Loaded {X.shape[0]} images (REAL={len(real_y)}, FAKE={len(fake_y)})")
    return X, y

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20, width_shift_range=0.2,
    height_shift_range=0.2, horizontal_flip=True, zoom_range=0.2
)

def augment_dataset(X, y, n_rounds=1, batch_size=32):
    datagen.fit(X)
    total = len(X) * n_rounds
    gen = datagen.flow(X, y, batch_size=batch_size, shuffle=True)
    X_aug, y_aug = [], []
    while len(X_aug)*batch_size < total:
        xb, yb = next(gen)
        X_aug.append(xb); y_aug.append(yb)
    X_aug = np.vstack(X_aug)[:total]
    y_aug = np.hstack(y_aug)[:total]
    X_comb = np.concatenate([X, X_aug], axis=0)
    y_comb = np.concatenate([y, y_aug], axis=0)
    print(f"Augmented dataset: {X_comb.shape[0]} samples ({len(X_aug)} augmented)")
    return X_comb, y_comb

def generate_adversarial_examples(
    model, X, y,
    attack_type="FGSM", eps=0.05,
    sample_frac=0.3, batch_size=2000):
    loss_obj = BinaryCrossentropy(label_smoothing=0.1)
    clsf = TensorFlowV2Classifier(model=model, nb_classes=2, input_shape=X.shape[1:], loss_object=loss_obj, clip_values=(-1.0,1.0))
    if attack_type == "FGSM":
        attack = FastGradientMethod(estimator=clsf, eps=eps)
    else:
        raise ValueError(f"Unknown attack: {attack_type}")

    n_adv = int(len(X) * sample_frac)
    idx = np.random.choice(len(X), n_adv, replace=False)
    X_src, y_src = X[idx], y[idx]

    adv_batches = []
    for i in range(0, n_adv, batch_size):
        j = min(i + batch_size, n_adv)
        print(f"[{attack_type}] Batch {i//batch_size+1}: {i}-{j-1}")
        adv_batches.append(attack.generate(x=X_src[i:j]))
    X_adv = np.concatenate(adv_batches)
    y_adv = y_src
    print(f"→ {attack_type} examples: {X_adv.shape[0]} samples")

    X_comb = np.concatenate([X, X_adv], axis=0)
    y_comb = np.concatenate([y, y_adv], axis=0)
    return X_comb, y_comb

def generate_fgsm_subset(model, X, y):
    loss_obj = BinaryCrossentropy(label_smoothing=0.1)
    clsf = TensorFlowV2Classifier(model=model, nb_classes=2, input_shape=(*config.IMG_SIZE,3), loss_object=loss_obj, clip_values=(-1.0,1.0))
    n = len(X)
    n_sub = int(n * (config.FGSM_FRAC))
    idx = np.random.choice(n, n_sub, replace=False)
    X_sub, y_sub = X[idx], y[idx]
    attack = FastGradientMethod(estimator=clsf, eps=config.FGSM_EPS)
    adv_batches = []
    for i in range(0, n_sub, config.BATCH_SIZE_ADV):
        j = min(i+ config.BATCH_SIZE_ADV, n_sub)
        print(f"[FGSM] Batch {i//config.BATCH_SIZE_ADV+1}: {i}-{j-1}")
        adv_batches.append(attack.generate(x=X_sub[i:j]))
    X_adv = np.concatenate(adv_batches, axis=0)
    return X_adv, y_sub