import numpy as np
from tqdm import tqdm
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path
import os
import joblib

# Ścieżki - dopasuj do swojego projektu
BASE_DIR = Path(__file__).parent.parent
CORE_DIR = BASE_DIR / "core"
DENOISED_DIR = BASE_DIR / "denoised_output"
DATASET_FILE = BASE_DIR / "dataset.npz"
SCALER_FILE = BASE_DIR / "scaler.pkl"
MODEL_FILE = BASE_DIR / "synth_model.keras"

sys.path.append(str(BASE_DIR))
from core.physisRT import PhysicalModelOrgan, synth_params_spec

SAMPLE_RATE = 44100
DURATION = 1.0
NUM_SAMPLES = 5000
TEST_SIZE = 0.2

# --- ekstrakcja cech, generate_dataset - jak wcześniej ---

def extract_adsr(audio, sr, threshold=0.05):
    envelope = np.abs(audio)
    envelope = envelope / np.max(envelope)
    attack_end = np.argmax(envelope > threshold)
    decay_end = attack_end
    for i in range(attack_end + 1, len(envelope)):
        if envelope[i] < envelope[i-1]:
            decay_end = i
            break
    release_start = len(envelope) - np.argmax(envelope[::-1] > threshold)
    if release_start <= decay_end:
        release_start = len(envelope)
    sustain_level = np.mean(envelope[decay_end:release_start]) if release_start > decay_end else 0
    attack_time = attack_end / sr
    decay_time = (decay_end - attack_end) / sr
    release_time = (len(envelope) - release_start) / sr
    return attack_time, decay_time, sustain_level, release_time

def extract_features(audio, sr):
    attack_time, decay_time, sustain_level, release_time = extract_adsr(audio, sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfcc, axis=1)
    features = np.zeros(17)
    features[0] = attack_time
    features[1] = decay_time
    features[2] = sustain_level
    features[3] = release_time
    features[4:] = mfcc_means
    return features

def generate_dataset(num_samples=NUM_SAMPLES, duration=DURATION, sample_rate=SAMPLE_RATE):
    X = np.zeros((num_samples, 17))
    param_names = [field[0] for field in synth_params_spec if field[0] not in ['sample_rate', 'block_size', 'epsilon']]
    y = np.zeros((num_samples, len(param_names)))
    organ = PhysicalModelOrgan()
    
    for i in tqdm(range(num_samples), desc="Generowanie danych"):
        random_params = {
            'CLIP1': np.random.uniform(0.1, 1.0),
            'CLIP2': np.random.uniform(0.1, 1.0),
            'GAIN1': np.random.uniform(0.1, 2.0),
            'GAIN2': np.random.uniform(0.1, 2.0),
            'GAIND': np.random.uniform(0.1, 1.0),
            'GAINF': np.random.uniform(0.1, 1.0),
            'CDEL': np.random.uniform(0.0, 0.5),
            'CBYP': np.random.uniform(0.0, 1.0),
            'X0': np.random.uniform(0.0, 0.5),
            'Y0': np.random.uniform(0.0, 0.5),
            'MOD_AMPL': np.random.uniform(0.0, 0.5),
            'NGAIN': np.random.uniform(0.1, 1.0),
            'NBFBK': np.random.uniform(0.0, 0.5),
            'NCGAIN': np.random.uniform(0.1, 1.0),
            'RATE_GAIN': np.random.uniform(0.5, 2.0),
            'NOISE_ATTACK': np.random.uniform(0.0, 0.1),
            'FBK': np.random.uniform(0.1, 1.0),
            'TFBK': np.random.uniform(0.5, 1.5),
            'RESONATOR_ATTACK': np.random.uniform(0.0, 0.1),
            'attack_time': np.random.uniform(0.01, 0.5),
            'decay_time': np.random.uniform(0.01, 0.5),
            'sustain_level': np.random.uniform(0.5, 1.0),
            'release_time': np.random.uniform(0.01, 0.5),
            'initial_level': 0.0
        }
        organ.set_params(random_params)
        audio = organ.generate(duration, freq=440.0)
        features = extract_features(audio, sample_rate)
        X[i] = features
        y[i] = np.array([random_params[name] for name in param_names])
    
    return X, y

def save_dataset(X, y, scaler):
    np.savez_compressed(DATASET_FILE, X=X, y=y)
    joblib.dump(scaler, SCALER_FILE)

def load_dataset():
    data = np.load(DATASET_FILE)
    scaler = joblib.load(SCALER_FILE)
    return data['X'], data['y'], scaler

def build_model(input_shape, output_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(output_shape)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def evaluate_model(model, X_test, y_test):
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Ocena modelu — Loss (MSE): {loss:.4f}, MAE: {mae:.4f}")

def auto_tune_synth(organ, target_wav_path, model, scaler):
    try:
        target_audio, sr = librosa.load(target_wav_path, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"Błąd wczytywania pliku: {e}")
        return
    features = extract_features(target_audio, sr)
    features = scaler.transform(features.reshape(1, -1))
    predicted_params = model.predict(features, verbose=0)[0]
    param_names = [field[0] for field in synth_params_spec if field[0] not in ['sample_rate', 'block_size', 'epsilon']]
    organ.set_params(dict(zip(param_names, predicted_params)))
    print("✅ Syntezator dostrojony!")

def main():
    append = input("📦 Czy chcesz dogenerować dane do istniejącego zbioru? (t/n): ").strip().lower() == "t"

    if append and DATASET_FILE.exists():
        print("📥 Wczytywanie istniejących danych...")
        X_old, y_old, scaler = load_dataset()
        X_new, y_new = generate_dataset()
        X_all = np.vstack((X_old, X_new))
        y_all = np.vstack((y_old, y_new))
    elif not DATASET_FILE.exists():
        print("🔨 Tworzenie nowego zbioru danych...")
        X_all, y_all = generate_dataset()
        scaler = StandardScaler()
    else:
        print("📥 Wczytywanie danych bez dogenerowania...")
        X_all, y_all, scaler = load_dataset()
        # Sprawdź czy jest zapisany model
        if MODEL_FILE.exists():
            print("💾 Ładowanie wytrenowanego modelu...")
            model = tf.keras.models.load_model(str(MODEL_FILE))
        else:
            print("🚂 Trening modelu od nowa...")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_all)
            save_dataset(X_scaled, y_all, scaler)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_all, test_size=TEST_SIZE)
            model = build_model((X_train.shape[1],), y_train.shape[1])
            model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)
            print("💾 Zapis modelu...")
            model.save(str(MODEL_FILE))
            evaluate_model(model, X_test, y_test)

        organ = PhysicalModelOrgan()
        auto_tune_synth(organ, str(DENOISED_DIR / "069-a_denoised.wav"), model, scaler)
        tuned_audio = organ.generate(8.0)
        organ.export(str(BASE_DIR / "tuned_output.wav"), tuned_audio)
        return

    print("📊 Skalowanie danych...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    save_dataset(X_scaled, y_all, scaler)

    print("🚂 Trening modelu...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_all, test_size=TEST_SIZE)
    model = build_model((X_train.shape[1],), y_train.shape[1])
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    print("💾 Zapis modelu...")
    model.save(str(MODEL_FILE))
    evaluate_model(model, X_test, y_test)

    organ = PhysicalModelOrgan()
    auto_tune_synth(organ, str(DENOISED_DIR / "069-a_denoised.wav"), model, scaler)
    tuned_audio = organ.generate(8.0)
    organ.export(str(BASE_DIR / "tuned_output.wav"), tuned_audio)

if __name__ == "__main__":
    main()
