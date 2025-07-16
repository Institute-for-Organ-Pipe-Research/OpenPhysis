import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wav


class PhysicalModelOrganRT:
    def __init__(self, sample_rate=44100, block_size=64):
        self.sample_rate = sample_rate
        self.block_size = block_size

        self.harmonic_gen = HarmonicGenerator(sample_rate)
        self.noise_gen = NoiseGenerator(sample_rate)
        self.resonator = LinearResonator(sample_rate)
        self.env_gen = EnvelopeGenerator(sample_rate)

        if params is None:
            self.params = self.default_params()
        else:
            self.params = params

        self.note_active = False
        self.note_freq = 0.0
        self.note_age = 0

    def default_params(self):
        return {
            'CLIP1': 0.7, 'CLIP2': 0.5, 'GAIN1': 1.0, 'GAIN2': 0.8,
            'GAIND': 0.6, 'GAINF': 0.4, 'CDEL': 0.3, 'CBYP': 0.7,
            'X0': 0.1, 'Y0': 0.05,
            'MOD_AMPL': 0.1,
            'NGAIN': 0.5, 'NBFBK': 0.4, 'NCGAIN': 0.6,
            'RATE_GAIN': 1.2, 'NOISE_ATTACK': 0.01,
            'FBK': 0.85, 'TFBK': 1.0, 'RESONATOR_ATTACK': 0.05,
            'attack_time': 0.1, 'decay_time': 0.05, 'sustain_level': 0.8,
            'release_time': 0.3, 'initial_level': 0.0,
            'epsilon': 1e-5
        }

    def set_params(self, params):
        self.params = params

    def note_on(self, freq):
        self.note_active = True
        self.note_freq = freq
        self.note_age = 0
        # Reset osc i innych stanów, jeśli trzeba
        self.harmonic_gen.osc.reset()
        self.noise_gen.reset()
        self.resonator.reset()

    def note_off(self):
        self.note_active = False

    def process_block(self):
        if not self.note_active:
            return np.zeros(self.block_size, dtype=np.float32)

        num_samples = self.block_size
        freq = self.note_freq
        start_index = self.note_age

        harmonic = self.harmonic_gen.generate(freq, num_samples, self.params, start_index)
        noise = self.noise_gen.generate(harmonic, num_samples, self.params, start_index)
        output = self.resonator.process(harmonic, noise, self.params, num_samples)

        # Usunięcie DC i normalizacja bloku
        output = output - np.mean(output)
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val

        self.note_age += num_samples
        return output

    def render_to_wav(self, freq, duration_sec, filename):
        self.note_on(freq)
        total_samples = int(self.sample_rate * duration_sec)
        blocks = []

        blocks_count = total_samples // self.block_size
        remainder = total_samples % self.block_size

        for _ in range(blocks_count):
            block = self.process_block()
            blocks.append(block)
        if remainder > 0:
            old_block_size = self.block_size
            self.block_size = remainder
            block = self.process_block()
            blocks.append(block)
            self.block_size = old_block_size

        self.note_off()

        audio = np.concatenate(blocks)
        audio = audio - np.mean(audio)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        audio_int16 = (audio * 32767).astype(np.int16)

        wav.write(filename, self.sample_rate, audio_int16)
        print(f"Zapisano plik WAV: {filename}")


class HarmonicGenerator:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.osc = HarmonicOscillator(sample_rate)
        self.env_gen = EnvelopeGenerator(sample_rate)
        self.freq_modulator = FrequencyModulator(sample_rate)
        self.lfo = LowFrequencyOscillator(sample_rate)
        self.delay_line = np.zeros(1024)
        self.delay_index = 0
        self.bp_b, self.bp_a = None, None

    def generate(self, freq, num_samples, params, start_index=0):
        output = np.zeros(num_samples)

        for i in range(num_samples):
            idx = start_index + i

            mod_freq = self.freq_modulator.process(freq, self.osc.var1)

            lfo_amp, lfo_freq = self.lfo.process()

            sin_wave = self.osc.process(mod_freq, params)

            double_freq = 2 * sin_wave**2 - 1

            path1 = sin_wave * params['GAIN1']
            path1 = np.clip(path1, -params['CLIP1'], params['CLIP1'])

            path2 = double_freq * params['GAIN2']
            path2 = np.clip(path2, -params['CLIP2'], params['CLIP2'])

            env = self.env_gen.attack_sustain_release(idx, int(params['attack_time'] * self.sample_rate * 10), params)
            path1 *= env
            path2 *= env

            sum_node = path1 + path2
            modulated = sum_node * (1 + params['MOD_AMPL'] * lfo_amp)

            delayed = self.delay_line[self.delay_index]
            filtered = params['CBYP'] * modulated + params['CDEL'] * delayed

            nonlin_in = filtered + params['X0']
            nonlin_out = nonlin_in - nonlin_in**4 + params['Y0']

            if self.bp_b is None or self.bp_a is None:
                low = max(20, 0.9 * freq)
                high = min(self.sample_rate / 2 - 1, 1.1 * freq)
                self.bp_b, self.bp_a = sig.butter(2, [low, high], btype='bandpass', fs=self.sample_rate)

            filtered_bp = sig.lfilter(self.bp_b, self.bp_a, [nonlin_out])[0]

            output[i] = params['GAIND'] * nonlin_out + params['GAINF'] * filtered_bp

            self.delay_line[self.delay_index] = modulated
            self.delay_index = (self.delay_index + 1) % len(self.delay_line)

        return output


class HarmonicOscillator:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.var1 = 1.0
        self.var2 = 0.0

    def reset(self):
        self.var1 = 1.0
        self.var2 = 0.0

    def process(self, freq, params):
        epsilon = params.get('epsilon', 1e-5)
        F = 2 * np.sin(np.pi * freq / self.sample_rate)

        new_var1 = self.var1 - F**2 * self.var2
        new_var2 = self.var2 * (1 + epsilon) + new_var1
        new_var1 = np.clip(new_var1, -1, 1)

        self.var1 = new_var1
        self.var2 = new_var2

        return self.var1


class FrequencyModulator:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.random_pitch = 1.0
        self.last_var1 = 0

    def process(self, base_freq, current_var1):
        if self.last_var1 < 0 <= current_var1:
            self.random_pitch = np.random.uniform(0.98, 1.02)
        self.last_var1 = current_var1
        return base_freq * self.random_pitch


class LowFrequencyOscillator:
    def __init__(self, sample_rate, freq=0.5, ampl=0.1, offset=0.0):
        self.sample_rate = sample_rate
        self.phase = 0.0
        self.freq = freq
        self.ampl = ampl
        self.offset = offset

    def process(self):
        self.phase += 2 * np.pi * self.freq / self.sample_rate
        self.phase %= 2 * np.pi

        triangle = 2 * np.abs(self.phase / np.pi - 1) - 1
        output_amp = self.offset + self.ampl * triangle

        if self.phase < np.pi:
            parabola = 2 * (self.phase / np.pi) ** 2 - 1
        else:
            parabola = 1 - 2 * ((self.phase - np.pi) / np.pi) ** 2
        output_freq = parabola

        return output_amp, output_freq


class NoiseGenerator:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.rate_limiter = RateLimiter()
        self.env_gen = EnvelopeGenerator(sample_rate)
        self.delay_lines = [np.zeros(100) for _ in range(4)]
        self.delay_indices = [0, 0, 0, 0]
        self.prev_limiter_out = 0.0
        self.lp_b, self.lp_a = sig.butter(2, 2000, 'lowpass', fs=self.sample_rate)

    def reset(self):
        self.delay_lines = [np.zeros(100) for _ in range(4)]
        self.delay_indices = [0, 0, 0, 0]
        self.prev_limiter_out = 0.0

    def generate(self, harmonic_signal, num_samples, params, start_index=0):
        output = np.zeros(num_samples)

        rate_signal = self._generate_rate_signal(harmonic_signal, params, num_samples)

        for i in range(num_samples):
            idx = start_index + i

            white_noise = np.random.uniform(-1, 1)
            filtered_noise = sig.lfilter(self.lp_b, self.lp_a, [white_noise])[0]

            node1 = filtered_noise + params['NBFBK'] * self.delay_lines[3][self.delay_indices[3]]
            self.delay_lines[0][self.delay_indices[0]] = node1
            delayed1 = self.delay_lines[0][self.delay_indices[0]]

            scaled = params['NCGAIN'] * delayed1

            node2 = scaled + self.delay_lines[1][self.delay_indices[1]]
            self.delay_lines[1][self.delay_indices[1]] = node2
            delayed2 = self.delay_lines[1][self.delay_indices[1]]

            self.delay_lines[2][self.delay_indices[2]] = delayed2
            delayed3 = self.delay_lines[2][self.delay_indices[2]]

            node3 = delayed2 + delayed3

            limited = self.rate_limiter.process(node3, rate_signal[i], self.prev_limiter_out)
            self.prev_limiter_out = limited

            self.delay_lines[3][self.delay_indices[3]] = limited
            delayed4 = self.delay_lines[3][self.delay_indices[3]]

            for j in range(4):
                self.delay_indices[j] = (self.delay_indices[j] + 1) % len(self.delay_lines[j])

            envelope = self.env_gen.noise_envelope(idx, int(params['NOISE_ATTACK'] * self.sample_rate * 10), params)

            output[i] = params['NGAIN'] * limited * envelope

        return output

    def _generate_rate_signal(self, harmonic, params, num_samples):
        rate = params['RATE_GAIN'] * harmonic
        rate = np.clip(rate, -1, 1)

        b, a = sig.butter(1, 100, 'highpass', fs=self.sample_rate)
        rate = sig.lfilter(b, a, rate)

        rate = np.maximum(rate, 0)

        attack_samples = int(params['NOISE_ATTACK'] * self.sample_rate)
        # Obwiednia dla aktualnego bloku: długość = num_samples
        envelope = np.ones(num_samples)
        if attack_samples > 0:
            attack_len = min(attack_samples, num_samples)
            envelope[:attack_len] = np.linspace(0, 1, attack_len)

        return rate * envelope



class RateLimiter:
    def process(self, input_val, rate_limit, prev_out):
        diff = input_val - prev_out
        diff_clipped = np.clip(diff, -rate_limit, rate_limit)
        return prev_out + diff_clipped


class LinearResonator:
    def __init__(self, sample_rate, buffer_size=2048):
        self.sample_rate = sample_rate
        self.delay_line = np.zeros(buffer_size)
        self.delay_index = 0
        self.apf_state = np.zeros(2)
        self.lp_b, self.lp_a = sig.butter(2, 4000, 'lowpass', fs=self.sample_rate)
        self.hp_b, self.hp_a = sig.butter(2, 50, 'highpass', fs=self.sample_rate)

    def reset(self):
        self.delay_line[:] = 0
        self.delay_index = 0
        self.apf_state[:] = 0

    def process(self, harmonic_signal, noise_signal, params, num_samples):
        output = np.zeros(num_samples)
        combined_input = harmonic_signal + noise_signal

        for i in range(num_samples):
            delay_output = self.delay_line[self.delay_index]

            lp_filtered = sig.lfilter(self.lp_b, self.lp_a, [delay_output])[0]
            hp_filtered = sig.lfilter(self.hp_b, self.hp_a, [lp_filtered])[0]

            if i < params['RESONATOR_ATTACK'] * self.sample_rate:
                fb_gain = params['FBK'] * 1.2
            else:
                fb_gain = params['FBK']

            apf_output = 0.7 * hp_filtered + self.apf_state[0]
            self.apf_state[0] = hp_filtered - 0.7 * apf_output

            feedback = params['TFBK'] * apf_output * fb_gain

            input_sum = combined_input[i] + feedback

            self.delay_line[self.delay_index] = input_sum

            output[i] = delay_output

            self.delay_index = (self.delay_index + 1) % len(self.delay_line)

        return output


class EnvelopeGenerator:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def attack_sustain_release(self, sample_index, total_samples, params):
        attack_time = params['attack_time']
        decay_time = params['decay_time']
        sustain_level = params['sustain_level']
        release_time = params['release_time']
        initial_level = params['initial_level']

        attack_end = int(attack_time * self.sample_rate)
        decay_end = attack_end + int(decay_time * self.sample_rate)
        release_start = total_samples - int(release_time * self.sample_rate)

        if sample_index < attack_end:
            return initial_level + (1.0 - initial_level) * (sample_index / attack_end)
        elif sample_index < decay_end:
            decay_pos = (sample_index - attack_end) / (decay_end - attack_end)
            return 1.0 - (1.0 - sustain_level) * decay_pos
        elif sample_index < release_start:
            return sustain_level
        else:
            release_pos = (sample_index - release_start) / (total_samples - release_start)
            return sustain_level * (1 - release_pos)

    def noise_envelope(self, sample_index, total_samples, params):
        attack_time = params['NOISE_ATTACK']
        attack_samples = int(attack_time * self.sample_rate)
        if sample_index < attack_samples:
            return sample_index / attack_samples
        return 1.0


if __name__ == "__main__":
    params = {
        'CLIP1': 0.7, 'CLIP2': 0.5, 'GAIN1': 1.0, 'GAIN2': 0.8,
        'GAIND': 0.6, 'GAINF': 0.4, 'CDEL': 0.3, 'CBYP': 0.7,
        'X0': 0.1, 'Y0': 0.05,
        'MOD_AMPL': 0.1,
        'NGAIN': 0.5, 'NBFBK': 0.4, 'NCGAIN': 0.6,
        'RATE_GAIN': 1.2, 'NOISE_ATTACK': 0.01,
        'FBK': 0.85, 'TFBK': 1.0, 'RESONATOR_ATTACK': 0.05,
        'attack_time': 0.1, 'decay_time': 0.05, 'sustain_level': 0.8,
        'release_time': 0.3, 'initial_level': 0.0,
        'epsilon': 1e-5
    }


    organ_rt = PhysicalModelOrganRT(sample_rate=44100, block_size=64)
    organ_rt.set_params(params)
    organ_rt.render_to_wav(freq=440.0, duration_sec=3.0, filename="organ_note.wav")

