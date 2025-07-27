import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import librosa
from pathlib import Path
from core.physisRT import PhysicalModelOrgan
from .future_extractor import extract_features
from .parameter_ranges import PARAMETER_RANGES

FEATURES_FILE = "generated_data.csv"

def generate_data(duration=1.0, sr=44100, max_samples=5000):
    """Generuje dane treningowe dla modelu organów"""
    X, Y = [], []
    organ = PhysicalModelOrgan()  # Inicjalizacja z domyślnymi parametrami
    param_names = list(PARAMETER_RANGES.keys())
    
    # Ustawienie sample_rate (jeśli klasa to obsługuje)
    if hasattr(organ.params, 'sample_rate'):
        organ.params.sample_rate = sr
    else:
        print("Uwaga: Klasa nie obsługuje bezpośredniego ustawienia sample_rate")

    print(f"Generuję {max_samples} próbek...")
    for _ in tqdm(range(max_samples), desc="Generowanie danych"):
        try:
            # 1. Losowanie parametrów
            params = {
                key: np.round(np.random.uniform(start, stop), 3)
                for key, (start, stop, _) in PARAMETER_RANGES.items()
            }
            
            # 2. Ustawienie parametrów i generacja dźwięku
            organ.set_params(params)
            signal = organ.generate(duration, freq=440.0)
            
            # 3. Ekstrakcja cech (z walidacją)
            feats = extract_features(signal, sr)
            if len(feats) != 33:
                raise ValueError(f"Otrzymano {len(feats)} cech zamiast 33")
                
            X.append(feats)
            Y.append([params[name] for name in param_names])
            
        except Exception as e:
            print(f"\nBłąd podczas generowania próbki: {str(e)}")
            continue

    # Konwersja i zapis danych
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    
    print("\nPodsumowanie danych:")
    print(f"- X shape: {X.shape} (dtype: {X.dtype})")
    print(f"- Y shape: {Y.shape} (dtype: {Y.dtype})")
    
    # Zapis do CSV
    df_X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(33)])
    df_Y = pd.DataFrame(Y, columns=param_names)
    df = pd.concat([df_X, df_Y], axis=1)
    df.to_csv(FEATURES_FILE, index=False)
    
    print(f"\nZapisano {len(X)} próbek do {FEATURES_FILE}")
    print(f"Struktura danych: {df.shape[1]} kolumn (33 cech + {len(param_names)} parametrów)")
    
    return X, Y

def build_model(input_dim=33, output_dim=None):
    if output_dim is None:
        output_dim = len(PARAMETER_RANGES)
        
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
    
    # Sprawdź czy plik zawiera odpowiednią liczbę cech
    n_params = len(PARAMETER_RANGES)
    expected_features = 33
    if df.shape[1] != expected_features + n_params:
        raise ValueError(f"Oczekiwano {expected_features + n_params} kolumn, otrzymano {df.shape[1]}. Wygeneruj nowe dane.")
    
    X = df.iloc[:, :expected_features].values.astype(np.float32)
    y = df.iloc[:, expected_features:].values.astype(np.float32)

    print("\n=== Dane treningowe ===")
    print(f"X shape: {X.shape}, dtype: {X.dtype}")
    print(f"y shape: {y.shape}, dtype: {y.dtype}")
    
    model = build_model(input_dim=expected_features)
    print("\n=== Architektura modelu ===")
    model.summary()
    
    model.fit(X, y, epochs=100, batch_size=64, validation_split=0.1)
    model.save(model_path)
    print(f"\nModel zapisany jako {model_path}")

def predict_from_audio(audio_path, model_path="synth_model.keras"):
    try:
        # 1. Wczytaj model
        model = tf.keras.models.load_model(model_path)
        
        # 2. Wczytaj audio
        signal, sr = librosa.load(audio_path, sr=44100)
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
            
        # 3. Ekstrakcja cech
        feats = extract_features(signal, sr)
        
        # 4. Walidacja
        if len(feats) != model.input_shape[1]:
            raise ValueError(
                f"Niezgodność cech: model oczekuje {model.input_shape[1]}, "
                f"otrzymano {len(feats)}. Sprawdź:\n"
                "1. Czy model był trenowany na 33 cechach?\n"
                "2. Czy extract_features() zwraca 33 wartości?"
            )
            
        # 5. Przygotowanie danych
        feats = np.array(feats, dtype=np.float32).reshape(1, -1)
        
        # 6. Predykcja
        pred = model.predict(feats, verbose=0)[0]
        return {k: float(v) for k, v in zip(PARAMETER_RANGES.keys(), pred)}
        
    except Exception as e:
        print(f"\n=== BŁĄD PREDYKCJI ===")
        print(f"Typ: {type(e).__name__}")
        print(f"Komunikat: {str(e)}")
        raise

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