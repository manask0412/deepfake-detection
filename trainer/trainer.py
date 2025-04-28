import os
from config import config
from tensorflow import keras
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from utils import plot_loss_acc, print_classification_report, plot_confusion_matrix, plot_roc

def build_model(input_shape=(32,32,3), dropout_rate=0.4):
    base = keras.applications.EfficientNetB2(
        include_top=False, weights="imagenet",
        input_shape=input_shape, pooling=None)
    base.trainable = False
    inp = keras.Input(shape=input_shape)
    x = base(inp, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out), base

def compile_and_train(
    model, base_model,
    X_tr, y_tr, X_val, y_val,
    phase_name, epochs=10, fine_tune=False, threshold=0.5):
    print(f"\n=== Phase: {phase_name!r} ===")
    if fine_tune:
        base_model.trainable = True

    lr = 1e-3 if not fine_tune else 1e-5
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=BinaryCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"])

    callbacks = [
        EarlyStopping("val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau("val_loss", factor=0.2, patience=2, min_lr=1e-6)]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=32,
        shuffle=True, verbose=1, callbacks=callbacks)

    phase_dir = os.path.join(config.OUTPUT_DIR, phase_name)
    os.makedirs(phase_dir, exist_ok=True)
    plot_loss_acc(history, os.path.join(phase_dir, "loss_acc_epoch.png"))

    y_val_probs = model.predict(X_val).flatten()
    y_val_pred = (y_val_probs >= threshold).astype(int)
    print_classification_report(y_val, y_val_pred)
    plot_confusion_matrix(y_val, y_val_pred, os.path.join(phase_dir, "confusion_matrix.png"))
    plot_roc(y_val, y_val_probs, os.path.join(phase_dir, "roc_curve.png"))

    return model, history