import time
import numpy as np
from numba import njit

class PhysicalModelOrgan:
    def __init__(self, sample_rate=44100, buffer_size=512):
        self.sample_rate = sample_rate
        self.params = self.default_params()

    def default_params(self):
        """Parametry zgodne z patentem z dodatkowymi ustawieniami"""
        return {
            # Generator harmoniczny
            'CLIP1': 0.7, 'CLIP2': 0.5, 'GAIN1': 1.0, 'GAIN2': 0.8,
            'GAIND': 0.6, 'GAINF': 0.4, 'CDEL': 0.3, 'CBYP': 0.7,
            'X0': 0.1, 'Y0': 0.05,  # Parametry nieliniowej funkcji
            'MOD_AMPL': 0.1,  # Głębokość modulacji amplitudy
            
            # Generator szumu
            'NGAIN': 0.5, 'NBFBK': 0.4, 'NCGAIN': 0.6,
            'RATE_GAIN': 1.2, 'NOISE_ATTACK': 0.01,
            
            # Rezonator
            'FBK': 0.85, 'TFBK': 1.0, 'RESONATOR_ATTACK': 0.05,
            
            # Obwiednie
            'attack_time': 0.1, 'decay_time': 0.05, 'sustain_level': 0.8,
            'release_time': 0.3, 'initial_level': 0.0
        }


class HarmonicGenerator:
    """Main harmonic sequence"""
    def __init__(self, sample_rate, buffer_size, params):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.osc = HarmonicOscillator(sample_rate, buffer_size)

    def process(self, freqs, params):
        if len(freqs) != self.buffer_size:
            raise ValueError("Rozmiar wektora częstotliwości niezgodny z buffer_size")

        # 1. Oscylator podstawowy – blok 16
        sin_wave = self.osc.process(freqs, params) # syg. 16

        # 2. Podwójna częstotliwość - blok 17
        double_freq = 2 * sin_wave**2 - 1 # syg. 17

        # Wzmocnienie GAIN1, GAIN2
        path1 = sin_wave * params['GAIN1'] # syg. 18a
        path1 = np.clip(path1, -params['CLIP1'], params['CLIP1']) # syg. 19a

        path2 = double_freq * params['GAIN2'] # syg. 18b
        path2 = np.clip(path2, -params['CLIP2'], params['CLIP2']) # syg. 19b


class HarmonicOscillator:
    """Blok 14 Sinusoidal oscilator"""
    def __init__(self, sample_rate=44100, buffer_size=512):
        self.sample_rate = 44100
        self.buffer_size = 512
        self.var1 = 1.0
        self.var2 = 0.0

    def no_numba_process(self, freqs, params):
            if len(freqs) != self.buffer_size:
                raise ValueError(f"Rozmiar bufora {len(freqs)} niezgodny z {self.buffer_size}")

            epsilon = params.get('epsilon', 1e-5)
            F = 2 * np.sin(np.pi * freqs / self.sample_rate)

            output = np.zeros_like(freqs)
            for i in range(self.buffer_size):
                self.var1 = self.var1 - F[i]**2 * self.var2
                self.var2 = self.var2 * (1 + epsilon)
                self.var2 = self.var1 + self.var2
                self.var1 = np.clip(self.var1, -1, 1)
                output[i] = self.var1

            return output
    
    def process(self, freqs, params):
        if len(freqs) != self.buffer_size:
            raise ValueError("Nieprawidłowy rozmiar bufora")
        
        epsilon = params.get("epsilon", 1e-5)
        F = 2 * np.sin(np.pi * freqs / self.sample_rate)

        output, self.var1, self.var2 = self.numba_process(
            self.var1, self.var2, F, epsilon
        )
        return output

    @staticmethod
    @njit
    def numba_process(var1, var2, F, epsilon):
        N = len(F)
        output = np.zeros(N)
        for i in range(N):
            var1 = var1 - F[i]**2 * var2
            var2 = var2 * (1 + epsilon)
            var2 = var1 + var2
            var1 = min(max(var1, -1.0), 1.0)
            output[i] = var1
        return output, var1, var2
    
