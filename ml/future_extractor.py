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
        print(f"Błąd w extract_adsr: {e}")
        return 0.05, 0.05, 0.7, 0.1

def extract_features(audio, sr):
    try:
        # 1. ADSR (4 cechy)
        attack, decay, sustain, release = extract_adsr(audio, sr)
        
        # 2. MFCC (13 cech)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfcc, axis=1)
        
        # 3. Chroma (12 cech)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_means = np.mean(chroma, axis=1)
        
        # 4. Inne cechy (4)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
        zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y=audio))
        rms = np.mean(librosa.feature.rms(y=audio))
        
        features = np.concatenate([
            [attack, decay, sustain, release],  # 4
            mfcc_means,                        # 13
            chroma_means,                      # 12
            [spectral_centroid, spectral_bandwidth, zero_crossing, rms]  # 4
        ])
        
        # Dodatkowe cechy jeśli potrzebne (do 33)
        if len(features) < 33:
            extra = np.zeros(33 - len(features))
            features = np.concatenate([features, extra])
            
        return features[:33]  # Zawsze 33 cechy
    
    except Exception as e:
        print(f"Błąd w extract_features: {e}")
        return np.zeros(33)  # Zwraca wektor zerowy o długości 33

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

def main():
    base_dir = Path(__file__).parent
    synthetic_dir = base_dir / "synthetic_data"
    acoustic_dir = base_dir / "acoustic_data"

    output_csv = base_dir / "extracted_features.csv"
    all_data = []

    # Przetwarzanie plików syntetycznych
    for wav_file in synthetic_dir.glob("*.wav"):
        audio, sr = librosa.load(str(wav_file), sr=44100)
        features = extract_features(audio, sr)
        all_data.append((wav_file.name, features, "synthetic"))

    # Przetwarzanie plików akustycznych
    for wav_file in acoustic_dir.rglob("*.wav"):
        audio, sr = librosa.load(str(wav_file), sr=44100)
        features = extract_features(audio, sr)
        # Dla voice_name proponuję użyć nazwy podfolderu jako głosu
        voice_name = wav_file.parent.name  
        all_data.append((wav_file.name, features, voice_name))

    export_features_to_csv(all_data, str(output_csv), append=False, include_voice_name=True)
    print(f"Zapisano cechy do {output_csv}")

if __name__ == "__main__":
    main()
