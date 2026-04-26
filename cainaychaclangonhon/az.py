import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ======================
# CONFIG
# ======================
DATA_DIR = "data"
SAMPLE_RATE = 16000
DURATION = 6
N_MFCC = 40

# ======================
# FEATURE EXTRACT
# ======================
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # padding / cắt
        expected_len = SAMPLE_RATE * DURATION
        if len(y) < expected_len:
            y = np.pad(y, (0, expected_len - len(y)))
        else:
            y = y[:expected_len]

        # ===== AUGMENT =====
        if np.random.rand() < 0.3:
            noise = np.random.randn(len(y))
            y = y + 0.005 * noise

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=N_MFCC
        )

        mfcc = np.mean(mfcc.T, axis=0)

        return mfcc

    except:
        return None

# ======================
# LOAD DATA
# ======================
X = []
y = []

for label, folder in enumerate(["not_cry", "cry"]):
    folder_path = os.path.join(DATA_DIR, folder)

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            path = os.path.join(folder_path, file)
            feat = extract_features(path)

            if feat is not None:
                X.append(feat)
                y.append(label)

X = np.array(X)
y = np.array(y)

print("Data shape:", X.shape)

# ======================
# SPLIT
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# CLASS WEIGHT
# ======================
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y),
    y=y
)

class_weights = dict(enumerate(weights))

# ======================
# MODEL (giảm overfit)
# ======================
model = Sequential([
    Dense(128, activation='relu', input_shape=(N_MFCC,)),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ======================
# CALLBACK
# ======================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# ======================
# TRAIN
# ======================
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[early_stop]
)

# ======================
# SAVE MODEL
# ======================
model.save("cry_model_best.h5")

print("✅ Training done!")