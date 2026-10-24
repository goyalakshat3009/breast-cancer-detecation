import random

import joblib
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

SEED = 3
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target  # target: 0 = malignant, 1 = benign

# stratify keeps the malignant/benign ratio the same in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2, stratify=y
)

# Fit the scaler on training data only (no leakage)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

model = keras.Sequential(
    [
        keras.layers.Input(shape=(30,)),
        keras.layers.Dense(20, activation="relu"),
        # softmax (not sigmoid) so the two outputs form a probability distribution
        keras.layers.Dense(2, activation="softmax"),
    ]
)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=8, restore_best_weights=True
)
model.fit(
    X_train_std,
    y_train,
    validation_split=0.1,
    epochs=100,
    callbacks=[early_stop],
    verbose=2,
)

loss, acc = model.evaluate(X_test_std, y_test, verbose=0)
print(f"\nTest accuracy: {acc:.4f}  loss: {loss:.4f}")

y_pred = np.argmax(model.predict(X_test_std, verbose=0), axis=1)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=data.target_names))

model.save("model.keras")
joblib.dump(scaler, "scaler.joblib")
print("Saved model.keras and scaler.joblib")