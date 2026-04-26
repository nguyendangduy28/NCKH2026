import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
import time

# ======================
# CONFIG
# ======================
SAMPLE_RATE = 16000
DURATION = 6
MODEL_PATH = "cry_modelJupyter.h5"
MAX_LEN = 180

# ======================
# LOAD MODEL
# ======================
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded!")

# ======================
# PREPROCESS (không cần lưu file nữa)
# ======================
def preprocess(audio):
    signal = audio.flatten()

    expected_len = SAMPLE_RATE * DURATION

    if len(signal) < expected_len:
        signal = np.pad(signal, (0, expected_len - len(signal)))
    else:
        signal = signal[:expected_len]

    mfcc = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=13)

    # fix length giống lúc train
    if mfcc.shape[1] < MAX_LEN:
        pad = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad)))
    else:
        mfcc = mfcc[:, :MAX_LEN]

    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

    mfcc = mfcc.T[..., np.newaxis]
    mfcc = mfcc[np.newaxis, ...]

    return mfcc

# ======================
# MAIN LOOP
# ======================
print("\n🎤 Listening continuously (6s window)... Ctrl+C to stop\n")

try:
    while True:
        print("Recording...")

        audio = sd.rec(int(SAMPLE_RATE * DURATION),
                       samplerate=SAMPLE_RATE,
                       channels=1)

        sd.wait()

        x = preprocess(audio)
        pred = model.predict(x, verbose=0)[0][0]

        if pred > 0.5:
            print(f"👶 Cry detected! ({pred:.2f})")
        else:
            print(f"😴 Not cry ({pred:.2f})")

        time.sleep(0.2)

except KeyboardInterrupt:
    print("\nStopped!")