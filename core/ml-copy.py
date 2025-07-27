import argparse
import numpy as np
import tensorflow as tf
from scipy.signal import hilbert
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from physical_model import PhysicalModelOrgan  # Twój model

FEATURES_FILE = "generated_data.csv"

# Lista parametrów do trenowania i ich zakresy (start, stop, step)
PARAMETER_RANGES = {
    'CLIP1': (0.1, 1.0, 0.01),
    'CLIP2': (0.1, 1.0, 0.01),
    'GAIND': (0.1, 1.0, 0.01),
    'GAINF': (0.1, 1.0, 0.01),
    'CDEL': (0.0, 10.0, 0.01),
    'CBYP': (0.0, 10.0, 0.01),
    'X0': (0.0, 1.0, 0.5),
    'Y0': (0.0, 1.0, 0.5),
    'MOD_AMPL': (0.0, 1.0, 0.1),
    'NGAIN': (0.0, 1.0, 0.01),
    'NBFBK': (0.0, 1.0, 0.01),
    'NCGAIN': (0.0, 1.0, 0.01),
    'RATE_GAIN': (0.0, 1.0, 0.01),
    'NOISE_ATTACK': (0.01, 0.5, 0.01),
    'FBK': (0.0, 0.9, 0.01),
    'TFBK': (0.0, 1.0, 0.01),
    'RESONATOR_ATTACK': (0.01, 0.5, 0.01),
    'attack_time': (0.01, 0.5, 0.01),
    'decay_time': (0.01, 1.0, 0.01),
    'sustain_level': (0.3, 1.0, 0.01),
    'release_time': (0.01, 5.0, 0.01),
    'initial_level': (0.0, 1.0, 0.01),
}

def extract_features(audio, sr):
    env = np.abs(hilbert(audio))
    mfcc = tf.signal.mfccs_from_log_mel_spectrograms(
        tf.math.log(tf.abs(tf.signal.stft(audio, frame_length=1024, frame_step=512)) + 1e-6)
    )
    mfcc = mfcc.numpy().mean(axis=0)
    feats = np.concatenate([
        [env.mean(), env.std(), np.max(env)],
        mfcc[:10]  # 10 współczynników MFCC
    ])
    return feats

def generate_data(duration=1.0, sr=44100):
    X, Y = [], []
    organ = PhysicalModelOrgan()
    param_names = list(PARAMETER_RANGES.keys())
    grids = [np.arange(*PARAMETER_RANGES[name]) for name in param_names]
    for combo in tqdm(np.array(np.meshgrid(*grids)).T.reshape(-1, len(param_names)),
                      desc="Generating combos"):
        params = dict(zip(param_names, combo))
        organ.set_params(params)
        audio = organ.generate(duration, freq=440.0)
        feats = extract_features(audio, sr)
        X.append(feats)
        Y.append(combo)
    df_X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(len(X[0]))])
    df_Y = pd.DataFrame(Y, columns=param_names)
    df = pd.concat([df_X, df_Y], axis=1)
    df.to_csv(FEATURES_FILE, index=False)
    print(f"Generated data saved to {FEATURES_FILE}")
    return df_X.values, df_Y.values

def build_model(input_dim, output_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_model(model_path="synth_model.keras"):
    df = pd.read_csv(FEATURES_FILE)
    n_params = len(PARAMETER_RANGES)
    X = df.iloc[:, :-(n_params)].values
    y = df.iloc[:, -n_params:].values
    model = build_model(X.shape[1], y.shape[1])
    model.fit(X, y, epochs=100, batch_size=32, validation_split=0.1)
    model.save(model_path)
    print(f"Model saved to {model_path}")

def predict_from_audio(audio, sr=44100, model_path="synth_model.keras"):
    model = tf.keras.models.load_model(model_path)
    feats = extract_features(audio, sr).reshape(1, -1)
    pred = model.predict(feats)[0]
    return dict(zip(PARAMETER_RANGES.keys(), pred))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['generate', 'train', 'predict'], required=True)
    p.add_argument('--audio', type=str, help="Path to WAV for predict")
    args = p.parse_args()

    if args.mode == 'generate':
        generate_data()
    elif args.mode == 'train':
        train_model()
    elif args.mode == 'predict':
        if args.audio is None:
            p.error("Podaj --audio ścieżkę do pliku WAV")
        audio, sr = tf.audio.decode_wav(tf.io.read_file(args.audio))
        audio = np.squeeze(audio.numpy())
        preds = predict_from_audio(audio, sr.numpy())
        print("Predicted parameters:", preds)

if __name__ == "__main__":
    main()
