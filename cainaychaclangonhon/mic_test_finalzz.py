import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import time

# ================= CONFIG =================
SAMPLE_RATE = 16000
DURATION = 12            # giảm xuống để phản hồi nhanh hơn
N_MFCC = 40

CHUNK = 0.5
BUFFER_SIZE = int(SAMPLE_RATE * DURATION)
STEP_SIZE = int(SAMPLE_RATE * CHUNK)

MODEL_PATH = "cry_model_best.h5"

# ================= LOAD MODEL =================
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded")

# Detect output type
output_shape = model.output_shape
print("🔍 Output shape:", output_shape)

if output_shape[-1] == 1:
    MODE = "sigmoid"
    print("👉 Using SIGMOID (binary)")
elif output_shape[-1] == 2:
    MODE = "softmax"
    print("👉 Using SOFTMAX (2 classes)")
else:
    raise ValueError("❌ Model output không hợp lệ")

# ================= BUFFER =================
buffer = np.zeros(BUFFER_SIZE)

# ================= FEATURE =================
def extract_features(signal):
    mfcc = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    return np.mean(mfcc.T, axis=0)

# ================= CALLBACK =================
def callback(indata, frames, time_info, status):
    global buffer

    if status:
        print("⚠️", status)

    audio = indata[:, 0]

    buffer[:] = np.roll(buffer, -len(audio))
    buffer[-len(audio):] = audio

# ================= MAIN =================
print("🎤 Listening... (Ctrl+C để dừng)")

try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=STEP_SIZE,
        callback=callback
    ):
        while True:
            signal = buffer.copy()

            # tránh predict khi buffer chưa đủ dữ liệu
            if np.abs(signal).mean() < 0.001:
                time.sleep(0.5)
                continue

            feat = extract_features(signal)
            feat = np.expand_dims(feat, axis=0)

            pred = model.predict(feat, verbose=0)

            # ===== SIGMOID =====
            # if MODE == "sigmoid":
            prob = pred[0][0]
            label = "👶 Cry" if prob >= 0.7 else "😴 Not Cry"
            print(f"{label} | Prob: {prob:.2f}")

            # ===== SOFTMAX =====
            # elif MODE == "softmax":
            #     probs = pred[0]
            #     labels = ["😴 Not Cry", "👶 Cry"]

            #     idx = np.argmax(probs)
            #     label = labels[idx]

            #     print(f"{label} | Confidence: {probs[idx]:.2f}")

            time.sleep(1)  # giảm tần suất in kết quả

except KeyboardInterrupt:
    print("\n🛑 Stopped")