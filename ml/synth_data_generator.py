import numpy as np
import pandas as pd
from .parameter_ranges import PARAMETER_RANGES
from core.physisRT import PhysicalModelOrgan  # Twój model fizyczny
from .future_extractor import *  # Funkcja ekstrakcji cech

DURATION = 1.0
SR = 44100
NUM_SAMPLES = 1000  # liczba próbek do wygenerowania

def generate_synthetic_data(output_csv="synthetic_features.csv"):
    organ = PhysicalModelOrgan()
    param_names = list(PARAMETER_RANGES.keys())

    X_features = []
    Y_params = []

    print(f"Generowanie {NUM_SAMPLES} próbek danych syntetycznych...")

    for _ in range(NUM_SAMPLES):
        params = {}
        for key, (start, stop, _) in PARAMETER_RANGES.items():
            params[key] = np.random.uniform(start, stop)

        organ.set_params(params)
        audio = organ.generate(DURATION, freq=440.0)
        features = extract_features(audio, SR)

        X_features.append(features)
        Y_params.append([params[name] for name in param_names])

    df_features = pd.DataFrame(X_features, columns=[f"feat_{i}" for i in range(len(X_features[0]))])
    df_params = pd.DataFrame(Y_params, columns=param_names)
    df = pd.concat([df_features, df_params], axis=1)
    df.to_csv(output_csv, index=False)

    print(f"Dane syntetyczne zapisane do {output_csv}")

if __name__ == "__main__":
    generate_synthetic_data()
