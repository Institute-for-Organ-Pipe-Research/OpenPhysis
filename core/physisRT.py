import numpy as np
from numba import njit, prange, float64, int32
from numba.experimental import jitclass
import scipy.io.wavfile as wav
from typing import Dict, Optional, Tuple

# 1. Pełna specyfikacja parametrów
synth_params_spec = [
    # Parametry generatora harmonicznego
    ('CLIP1', float64), ('CLIP2', float64),
    ('GAIN1', float64), ('GAIN2', float64),
    ('GAIND', float64), ('GAINF', float64),
    ('CDEL', float64), ('CBYP', float64),
    ('X0', float64), ('Y0', float64),
    ('MOD_AMPL', float64),
    
    # Parametry generatora szumu
    ('NGAIN', float64), ('NBFBK', float64),
    ('NCGAIN', float64), ('RATE_GAIN', float64),
    ('NOISE_ATTACK', float64),
    
    # Parametry rezonatora
    ('FBK', float64), ('TFBK', float64),
    ('RESONATOR_ATTACK', float64),
    
    # Parametry obwiedni
    ('attack_time', float64), ('decay_time', float64),
    ('sustain_level', float64), ('release_time', float64),
    ('initial_level', float64),
    
    # Parametry globalne
    ('epsilon', float64),
    ('sample_rate', int32),
    ('block_size', int32)
]

@jitclass(synth_params_spec)
class SynthParameters:
    def __init__(self):
        # Inicjalizacja parametrów domyślnych
        self.CLIP1 = 0.7
        self.CLIP2 = 0.5
        self.GAIN1 = 1.0
        self.GAIN2 = 0.8
        self.GAIND = 0.6
        self.GAINF = 0.4
        self.CDEL = 0.3
        self.CBYP = 0.7
        self.X0 = 0.1
        self.Y0 = 0.05
        self.MOD_AMPL = 0.1
        
        self.NGAIN = 0.5
        self.NBFBK = 0.4
        self.NCGAIN = 0.6
        self.RATE_GAIN = 1.2
        self.NOISE_ATTACK = 0.01
        
        self.FBK = 0.85
        self.TFBK = 1.0
        self.RESONATOR_ATTACK = 0.05
        
        self.attack_time = 0.1
        self.decay_time = 0.05
        self.sustain_level = 0.8
        self.release_time = 0.3
        self.initial_level = 0.0
        
        self.epsilon = 1e-5
        self.sample_rate = 44100
        self.block_size = 64

class PhysicalModelOrgan:
    def __init__(self, params: Optional[Dict] = None):
        """
        Inicjalizacja modelu fizycznego organów
        """
        self.params = SynthParameters()
        if params:
            self.set_params(params)
        
        # Inicjalizacja buforów
        self._output_buffer = np.zeros(self.params.block_size)
        self._delay_line = np.zeros(1024)
        self._delay_index = 0
        self._var1 = 1.0  # Stan oscylatora
        self._var2 = 0.0  # Stan oscylatora
        
    def set_params(self, param_dict: Dict):
        """Ustawia parametry z słownika"""
        for key, value in param_dict.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
    
    def get_params(self) -> Dict:
        """Zwraca aktualne parametry jako słownik"""
        return {field: getattr(self.params, field) for field, _ in synth_params_spec}
    
    @staticmethod
    @njit
    def _process_oscillator(freq: float, var1: float, var2: float, 
                          sample_rate: float, epsilon: float) -> Tuple[float, float, float]:
        F = 2 * np.sin(np.pi * freq / sample_rate)
        new_var1 = var1 - F**2 * var2
        new_var2 = var2 * (1 + epsilon) + new_var1
        new_var1 = max(min(new_var1, 1.0), -1.0)
        return new_var1, new_var2, F
    
    @staticmethod
    @njit
    def _apply_nonlinearity(x: float, x0: float, y0: float) -> float:
        return (x + x0) - (x + x0)**4 + y0
    
    def process_block(self) -> np.ndarray:
        """
        Generuje blok dźwięku o rozmiarze block_size
        """
        p = self.params
        output = np.zeros(p.block_size)
        
        for i in range(p.block_size):
            # Generacja sygnału harmonicznego
            self._var1, self._var2, F = self._process_oscillator(
                440.0, self._var1, self._var2, p.sample_rate, p.epsilon)
            
            # Przetwarzanie nieliniowe
            nonlin_out = self._apply_nonlinearity(self._var1, p.X0, p.Y0)
            
            # Efekty i mix
            output[i] = nonlin_out * p.GAIND
        
        return output
    
    def generate(self, duration: float, freq: float = 440.0) -> np.ndarray:
        """
        Generuje dźwięk o podanym czasie trwania
        """
        num_blocks = int(np.ceil(duration * self.params.sample_rate / self.params.block_size))
        output = np.zeros(num_blocks * self.params.block_size)
        
        for i in range(num_blocks):
            block = self.process_block()
            output[i * self.params.block_size : (i + 1) * self.params.block_size] = block
        
        return output[:int(duration * self.params.sample_rate)]
    
    def export(self, filename: str, audio_data: Optional[np.ndarray] = None):
        """
        Eksportuje dźwięk do pliku WAV
        """
        data = audio_data if audio_data is not None else self._output_buffer
        if data is None:
            raise ValueError("Najpierw wygeneruj dźwięk używając generate()")
        
        data_int16 = (data * 32767).astype(np.int16)
        wav.write(filename, self.params.sample_rate, data_int16)
        print(f"Zapisano: {filename}")

# Przykład użycia
if __name__ == "__main__":
    # Inicjalizacja z domyślnymi parametrami
    organ = PhysicalModelOrgan()
    
    # Przykładowe parametry
    custom_params = {
        'CLIP1': 0.8,
        'GAIN1': 1.2,
        'NOISE_ATTACK': 0.02,
        'FBK': 0.9
    }
    organ.set_params(custom_params)
    
    # Generacja i eksport
    audio = organ.generate(3.0, 440.0)
    organ.export("organ_output.wav", audio)