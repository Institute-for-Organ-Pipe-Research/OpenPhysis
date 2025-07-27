from pathlib import Path
import numpy as np
import librosa
from scipy.signal import hilbert
import csv
import os

def extract_adsr(audio, sr, threshold=0.05):
    try:
        if len(audio) == 0:
            return 0.05, 0.05, 0.7, 0.1
            
        analytic_signal = hilbert(audio)
        envelope = np.abs(analytic_signal)
        envelope = envelope / (np.max(envelope) + 1e-6)
        
        above_threshold = envelope > threshold
        attack_end = np.argmax(above_threshold) if np.any(above_threshold) else len(envelope)//10
        
        decay_phase = envelope[attack_end:] < 0.7
        decay_end = attack_end + np.argmax(decay_phase) if np.any(decay_phase) else len(envelope)//2
        
        release_phase = envelope[::-1] > threshold
        release_start = len(envelope) - np.argmax(release_phase) if np.any(release_phase) else len(envelope)*9//10
        
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
        attack, decay, sustain, release = extract_adsr(audio, sr)
        
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfcc, axis=1)
        mfcc_deltas = np.mean(librosa.feature.delta(mfcc), axis=1)
        
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
        harmonic = librosa.effects.harmonic(y=audio)
        harmonic_ratio = np.mean(harmonic / (np.abs(audio) + 1e-6))
        
        features = np.concatenate([
            [attack, decay, sustain, release],
            mfcc_means,
            mfcc_deltas,
            [spectral_centroid, spectral_bandwidth, harmonic_ratio]
        ])
        
        if np.isnan(features).any():
            features = np.nan_to_num(features, nan=0.0)
            
        return features
        
    except Exception as e:
        print(f"Błąd w extract_features: {str(e)}")
        return np.zeros(33)

def export_features_to_csv(data, output_file, append=False, include_voice_name=False):
    file_exists = os.path.isfile(output_file)
    mode = 'a' if append else 'w'

    with open(output_file, mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        if not append or not file_exists:
            header = ['filename']
            if include_voice_name:
                header.append('voice_name')
            header += [f'feature_{i+1}' for i in range(len(data[0][1]))]
            writer.writerow(header)

        for record in data:
            if include_voice_name:
                filename, features, voice_name = record
                row = [filename, voice_name] + list(features)
            else:
                filename, features = record
                row = [filename] + list(features)
            writer.writerow(row)


def main(voice_name="default_voice"):
    BASE_DIR = Path(__file__).parent.parent
    DENOISED_DIR = BASE_DIR / "denoised_output"

    data_to_export = []

    for plik_audio in DENOISED_DIR.glob("*.wav"):
        audio, sr = librosa.load(str(plik_audio), sr=44100)
        features = extract_features(audio, sr)
        print(f"Przetwarzam: {plik_audio.name} - {len(features)} cech")
        data_to_export.append((plik_audio.name, features, voice_name))

    export_features_to_csv(data_to_export, "export_features.csv", append=False, include_voice_name=True)
    print("Zapisano cechy do export_features.csv")

if __name__ == "__main__":
    main("principal_8")
