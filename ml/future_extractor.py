import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from pathlib import Path
import librosa

class AudioSignal:
    def __init__(self):
        self.audio = None
        self.sr = None

    def open_audio(self, path_audio):
        self.audio, self.sr = librosa.load(path_audio, sr=None)

class ADSRExtractor:
    def __init__(self, path):
        self.signal = AudioSignal()
        self.signal.open_audio(path)
        self.envelope = self.compute_envelope()
        self.attack = self.detect_attack_end()
        self.decay = self.detect_decay_end(self.attack['i'])
        self.release = self.detect_release_start(self.decay['i'])
        self.sustain_level = self.calculate_sustain_level(self.decay['i'], self.release['i'])

    def compute_envelope(self):
        analytic_signal = scipy.signal.hilbert(self.signal.audio)
        envelope = np.abs(analytic_signal)
        # Normalizacja względem amplitudy sygnału audio
        norm_envelope = envelope / (np.max(np.abs(self.signal.audio)) + 1e-6)
        return norm_envelope

    def detect_attack_end(self, slope_threshold=-0.01, flat_threshold=0.002):
        sr = self.signal.sr
        slope = np.diff(self.envelope)
        max_idx = np.argmax(self.envelope)

        for i in range(max_idx, len(slope) - 10):
            window = slope[i:i+10]
            if np.all(np.abs(window) < flat_threshold):
                return {'i': i, 'time': i / sr}

        return {'i': -1, 'time': -1.0}

    def detect_decay_end(self, attack_end_idx, threshold_slope=0.005, min_decay_duration=10000, window_confirm=50):
        """
        Detekcja końca fazy decay oparta na pochodnych.
        """
        sr = self.signal.sr
        env = self.envelope
        slope = np.diff(env)

        start_idx = attack_end_idx + min_decay_duration
        end_idx = len(slope) - window_confirm

        for i in range(start_idx, end_idx):
            window = slope[i:i + window_confirm]
            if np.all(np.abs(window) < threshold_slope):
                return {'i': i, 'time': i / sr}

        # Jeżeli nie znaleziono końca decay
        return {'i': -1, 'time': -1.0}

    def detect_release_start(self, decay_end_idx, threshold_release=0.001, window_confirm=100):
        sr = self.signal.sr
        slope = np.diff(self.envelope)
        # Szukamy wstecz, od końca sygnału do decay_end_idx
        for i in range(len(slope) - window_confirm, decay_end_idx, -1):
            window = slope[i - window_confirm:i]
            if np.all(window < -threshold_release):
                return {'i': i, 'time': i / sr}
        return {'i': -1, 'time': -1.0}

    def calculate_sustain_level(self, decay_end_idx, release_start_idx):
        if decay_end_idx == -1:
            return -1
        if release_start_idx == -1:
            # Jeśli nie znaleziono release, sustain liczymy do końca sygnału
            sustain_region = self.envelope[decay_end_idx:]
        else:
            sustain_region = self.envelope[decay_end_idx:release_start_idx]

        if len(sustain_region) == 0:
            return -1
        return np.mean(sustain_region)


    def extract(self):
        return {
            "attack_idx": self.attack['i'],
            "attack_time": self.attack['time'],
            "decay_idx": self.decay['i'],
            "decay_time": self.decay['time'],
            "sustain_level": self.sustain_level,
            "release_idx": self.release['i'],
            "release_time": self.release['time']
        }

def plot_adsr(extractor):
    audio = extractor.signal.audio
    envelope = extractor.envelope
    sr = extractor.signal.sr
    times = np.arange(len(audio)) / sr

    plt.figure(figsize=(12, 6))
    plt.plot(times, envelope, label='Obwiednia', linewidth=2)

    if extractor.attack['i'] != -1:
        plt.axvline(extractor.attack['time'], color='green', linestyle='--', label=f'End attack: {extractor.attack["time"]:.3f}s')
    if extractor.decay['i'] != -1:
        plt.axvline(extractor.decay['time'], color='red', linestyle='--', label=f'End decay: {extractor.decay["time"]:.3f}s')
    if extractor.release['i'] != -1:
        plt.axvline(extractor.release['time'], color='purple', linestyle='--', label=f'Start release: {extractor.release["time"]:.3f}s')

    # Pokazujemy sustain jako poziomą linię
    if extractor.sustain_level != -1:
        plt.hlines(extractor.sustain_level, extractor.decay['time'], extractor.release['time'], colors='orange', linestyles='-', label='Sustain level')

    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.title("Obwiednia i fazy ADSR")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    base_dir = Path(__file__).parent
    acoustic_dir = base_dir / "acoustic_data/hauptwerk-principal8-001"
    wav_files = list(acoustic_dir.glob("*.wav")) + list(acoustic_dir.glob("*.WAV"))

    if not wav_files:
        print("Brak plików WAV!")
        return

    first_wav = wav_files[0]
    print(f"Testuję plik: {first_wav.name}")

    try:
        extractor = ADSRExtractor(str(first_wav))
        result = extractor.extract()
        print(f"Wynik extract(): {result}")
        plot_adsr(extractor)
    except Exception as e:
        print(f"Błąd podczas przetwarzania pliku {first_wav.name}: {e}")

if __name__ == "__main__":
    main()
