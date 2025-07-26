from pathlib import Path
import numpy as np
import librosa
from scipy.signal import hilbert

def extract_adsr(audio, sr, threshold=0.05):
    try:
        # Oblicz obwiednię używając transformaty Hilberta z scipy
        if len(audio) == 0:
            return 0.05, 0.05, 0.7, 0.1
            
        analytic_signal = hilbert(audio)
        envelope = np.abs(analytic_signal)
        envelope = envelope / (np.max(envelope) + 1e-6)  # Normalizacja z zabezpieczeniem
        
        # Znajdź punkty charakterystyczne z zabezpieczeniami
        above_threshold = envelope > threshold
        attack_end = np.argmax(above_threshold) if np.any(above_threshold) else len(envelope)//10
        
        decay_phase = envelope[attack_end:] < 0.7
        decay_end = attack_end + np.argmax(decay_phase) if np.any(decay_phase) else len(envelope)//2
        
        release_phase = envelope[::-1] > threshold
        release_start = len(envelope) - np.argmax(release_phase) if np.any(release_phase) else len(envelope)*9//10
        
        # Oblicz parametry ADSR
        attack_time = attack_end / sr
        decay_time = (decay_end - attack_end) / sr
        sustain_level = np.mean(envelope[decay_end:release_start]) if release_start > decay_end else 0.7
        release_time = (len(envelope) - release_start) / sr
        
        return max(0.01, attack_time), max(0.01, decay_time), np.clip(sustain_level, 0, 1), max(0.01, release_time)
        
    except Exception as e:
        print(f"Błąd w extract_adsr: {str(e)}")
        return 0.05, 0.05, 0.7, 0.1

def extract_features(audio, sr):
    try:
        # Ekstrakcja ADSR
        attack, decay, sustain, release = extract_adsr(audio, sr)
        
        # Ekstrakcja MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfcc, axis=1)
        mfcc_deltas = np.mean(librosa.feature.delta(mfcc), axis=1)
        
        # Cechy spektralne
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
        harmonic = librosa.effects.harmonic(y=audio)
        harmonic_ratio = np.mean(harmonic / (np.abs(audio) + 1e-6))
        
        # Tworzenie wektora cech
        features = np.concatenate([
            [attack, decay, sustain, release],
            mfcc_means,
            mfcc_deltas,
            [spectral_centroid, spectral_bandwidth, harmonic_ratio]
        ])
        
        # Sprawdź czy nie ma NaN
        if np.isnan(features).any():
            features = np.nan_to_num(features, nan=0.0)
            
        return features
        
    except Exception as e:
        print(f"Błąd w extract_features: {str(e)}")
        return np.zeros(33)  # 4 ADSR + 13 MFCC + 13 delt + 3 cechy spektralne

def extract_features(audio, sr):
    try:
        # Ekstrakcja ADSR
        adsr_values = extract_adsr(audio, sr)
        if len(adsr_values) != 4 or not all(isinstance(x, (int, float)) for x in adsr_values):
            adsr_values = (0.05, 0.05, 0.7, 0.1)  # wartości domyślne
        
        attack_time, decay_time, sustain_level, release_time = adsr_values
        
        # Ekstrakcja MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfcc, axis=1)
        mfcc_deltas = librosa.feature.delta(mfcc)
        
        # Ekstrakcja cech spektralnych
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
        harmonic = librosa.effects.harmonic(y=audio)
        harmonic_ratio = np.mean(harmonic / (np.abs(audio) + 1e-6))
        
        # Tworzenie wektora cech
        features = np.concatenate([
            [attack_time, decay_time, sustain_level, release_time],
            mfcc_means,
            np.mean(mfcc_deltas, axis=1),
            [spectral_centroid, spectral_bandwidth, harmonic_ratio]
        ])
        
        return features
        
    except Exception as e:
        print(f"Błąd w extract_features: {str(e)}")
        # Zwróć wektor zerowy o przewidywanym rozmiarze
        return np.zeros(13 + 13 + 3 + 4)  # 13 MFCC + 13 delt + 3 cechy spektralne + 4 ADSR
    
if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent
    CORE_DIR = BASE_DIR / "core"
    DENOISED_DIR = BASE_DIR / "denoised_output"
    DATASET_FILE = BASE_DIR / "dataset.npz"
    SCALER_FILE = BASE_DIR / "scaler.pkl"
    MODEL_FILE = BASE_DIR / "synth_model.keras"



    audio, sr = librosa.load(str(DENOISED_DIR / "069-a_denoised.wav"), sr=44100)
    features = extract_features(audio, sr)
    print(f"Wyekstrahowano {len(features)} cech")
    print(f"Przykładowe wartości: {features[:10]}")