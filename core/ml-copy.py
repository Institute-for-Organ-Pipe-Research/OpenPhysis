import argparse
import numpy as np
import tensorflow as tf
from scipy.signal import hilbert
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import soundfile as sf

from core.physisRT import PhysicalModelOrgan  # Upewnij się, że ścieżka się zgadza

FEATURES_FILE = "generated_data.csv"

def extract_features(audio, sr):
    """Wyciąganie cech: obwiednia + 10 MFCC"""
    env = np.abs(hilbert(audio))
    stft = tf.signal.stft(audio, frame_length=1024, frame_step=512)
    spectrogram = tf.abs(stft)
    mel = tf.tensordot(spectrogram, tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=40, num_spectrogram_bins=spectrogram.shape[-1], sample_rate=sr), 1)
    log_mel = tf.math.log(mel + 1e-6)
    mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., :10]
    mfcc = tf.reduce_mean(mfcc, axis=0).numpy()
    feats = np.concatenate([
        [env.mean(), env.std(), np.max(env)],
        mfcc
    ])
    return feats

def generate_data(duration=1.0, sr=44100, max_samples=5000):
    X, Y = [], []
    organ = PhysicalModelOrgan(sample_rate=sr)
    param_names = list(PARAMETER_RANGES.keys())

    # Losowanie kombinacji (aby nie zalać RAM)
    print("Generuję dane losowo...")
    for _ in tqdm(range(max_samples)):
        params = {}
        for key, (start, stop, step) in PARAMETER_RANGES.items():
            value = np.round(np.random.uniform(start, stop), 3)
            params[key] = value
        organ.set_params(params)
        signal = organ.generate(duration, freq=440.0)
        feats = extract_features(signal, sr)
        X.append(feats)
        Y.append(list(params.values()))

    X = np.array(X)
    Y = np.array(Y)
    df_X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    df_Y = pd.DataFrame(Y, columns=param_names)
    df = pd.concat([df_X, df_Y], axis=1)
    df.to_csv(FEATURES_FILE, index=False)
    print(f"Zapisano {len(X)} przykładów do {FEATURES_FILE}")
    return X, Y

def build_model(input_dim, output_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(output_dim)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_model(model_path="synth_model.keras"):
    df = pd.read_csv(FEATURES_FILE)
    n_params = len(PARAMETER_RANGES)
    X = df.iloc[:, :-(n_params)].values
    y = df.iloc[:, -n_params:].values
    model = build_model(X.shape[1], y.shape[1])
    model.fit(X, y, epochs=100, batch_size=64, validation_split=0.1)
    model.save(model_path)
    print(f"Model zapisany jako {model_path}")

def predict_from_audio(audio_path, model_path="synth_model.keras"):
    model = tf.keras.models.load_model(model_path)
    signal, sr = sf.read(audio_path)
    if signal.ndim > 1:  # stereo
        signal = signal.mean(axis=1)
    feats = extract_features(signal, sr).reshape(1, -1)
    pred = model.predict(feats)[0]
    return dict(zip(PARAMETER_RANGES.keys(), pred))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['generate', 'train', 'predict'], required=True)
    p.add_argument('--audio', type=str, help="Ścieżka do pliku WAV")
    args = p.parse_args()

    if args.mode == 'generate':
        generate_data()
    elif args.mode == 'train':
        train_model()
    elif args.mode == 'predict':
        if not args.audio:
            raise ValueError("Użyj --audio z plikiem WAV")
        result = predict_from_audio(args.audio)
        print("Przewidziane parametry:", result)

if __name__ == "__main__":
    main()
