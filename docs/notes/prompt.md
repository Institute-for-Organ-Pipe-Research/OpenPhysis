Opracowuje implementacje Physis według patentu https://patents.google.com/patent/US7442869B2/en. Jestem na etapie analizy `Main Harmonic Sequence`. Do każdego kluczowego bloku lub bloków tworze wykresy w matplotlib z interaktywnymi suwakami.


```myrmaind
classDiagram
    class PhysicalModelOrgan {
        +sample_rate: int
        +harmonic_gen: HarmonicGenerator
        +noise_gen: NoiseGenerator
        +resonator: LinearResonator
        +render_note(freq, duration): np.ndarray
    }
    
    class HarmonicGenerator {
        +osc: HarmonicOscillator
        +env_gen: EnvelopeGenerator
        +freq_modulator: FrequencyModulator
        +generate(freq, duration): np.ndarray
    }
    
    class HarmonicOscillator {
        +var1, var2: float
        +process(freq): float
    }
    
    class NoiseGenerator {
        +rate_limiter: RateLimiter
        +generate(harmonic_signal): np.ndarray
    }
    
    class LinearResonator {
        +process(harmonic, noise): np.ndarray
    }
    
    PhysicalModelOrgan *-- HarmonicGenerator
    PhysicalModelOrgan *-- NoiseGenerator
    PhysicalModelOrgan *-- LinearResonator
    HarmonicGenerator *-- HarmonicOscillator
    HarmonicGenerator *-- EnvelopeGenerator
```