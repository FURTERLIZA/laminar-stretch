#!/usr/bin/env python3
"""
laminar_stretch.py -- laminar stretch

granular audio time-stretcher. the source is scanned slowly (hop_in << hop_out)
and windowed segments are overlap-added into the output. each output grain is a
Hann-weighted blend of several nearby source grains, placed at a jittered position.
multiple independent layers, each with different grain size, stretch modifier, and
pitch offset, are mixed to produce the output.

optional post-processing: reverb, compression, tone correction, pitch analysis / MIDI.

usage:
    python laminar_stretch.py input.wav output.wav --stretch 800
    python laminar_stretch.py input.wav output.wav --stretch 200 --num-layers 2
    python laminar_stretch.py input.wav output.wav --stretch 400 --grain-ms 200 --seed 42
    python laminar_stretch.py input.wav output.wav --stretch 800 --reverb --reverb-room 0.8
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io.wavfile as wavfile
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import butter, lfilter, oaconvolve


# ---------------------------------------------------------------------------
#  utilities
# ---------------------------------------------------------------------------

def cents_to_ratio(cents: float) -> float:
    """convert a pitch offset in cents to a linear frequency ratio."""
    return 2.0 ** (cents / 1200.0)


# ---------------------------------------------------------------------------
#  layer configuration
# ---------------------------------------------------------------------------

@dataclass
class LayerConfig:
    """
    parameters for one layer.

    stretch_modifier
        multiplied by the global stretch factor for this layer only.
        values < 1 produce a shorter layer that drifts out of phase
        with the foundation over time.

    pitch_ratio
        > 1.0 scans the source faster, raising perceived pitch.
        < 1.0 scans slower, lowering it.
        hop_in = hop_out * pitch_ratio / eff_stretch, so it is
        independent of the time stretch. small offsets (+-7-15 cents)
        produce a chorus effect.

    grain_ms
        grain window duration in milliseconds. when auto_grain is on,
        this is scaled up at extreme stretch ratios.

    overlap
        fraction of grain_size by which consecutive grains overlap.
        hop_out = grain_size * (1 - overlap).

    jitter_ms
        maximum random offset (+-) applied to each grain's source position.

    n_avg
        number of adjacent source grains blended into each output grain.
        1 = no blending. 3-4 = centre-weighted average.

    avg_spread
        neighbourhood radius for averaging, as a fraction of grain_size.

    gain
        contribution of this layer in the final mix (before peak normalisation).

    phase_rand_ms
        maximum additional random offset (+-ms) applied to the source position
        after jitter. finer than jitter and independent per grain; reduces
        inter-layer phase coherence. set to 0 to disable.
    """
    stretch_modifier: float = 1.0
    pitch_ratio:      float = 1.0
    grain_ms:         float = 150.0
    overlap:          float = 0.75
    jitter_ms:        float = 20.0
    n_avg:            int   = 3
    avg_spread:       float = 0.5
    gain:             float = 1.0
    phase_rand_ms:    float = 8.0


# ---------------------------------------------------------------------------
#  LFO
# ---------------------------------------------------------------------------

@dataclass
class LFO:
    """
    low-frequency oscillator that modulates one parameter of every layer.

    rate_hz   modulation frequency in Hz.

    depth     proportion of the base parameter value displaced by the LFO.
              0.3 = parameter swings +-30% around its base value.

    target    parameter to modulate:
                "grain_ms"   -- grain window size
                "jitter_ms"  -- source-position scatter
                "gain"       -- layer amplitude (applied post-loop)

    waveform  shape of the modulation signal:
                "sine"         -- sinusoidal
                "triangle"     -- linear ramp up/down
                "random_walk"  -- white noise LP-filtered to rate_hz
    """
    rate_hz:  float = 0.08
    depth:    float = 0.3
    target:   str   = "grain_ms"    # "grain_ms" | "jitter_ms" | "gain"
    waveform: str   = "sine"        # "sine" | "triangle" | "random_walk"


# ---------------------------------------------------------------------------
#  chord mode
# ---------------------------------------------------------------------------

@dataclass
class ChordMode:
    """
    copies the layer stack for each semitone interval and mixes them.

    interval 0 is the root and uses standard layer gains. other intervals
    receive a copy with pitch_ratio multiplied by 2^(semitones/12), mixed at gain.

    presets: fifth [0,7]  octave [0,12]  minor [0,3,7]  major [0,4,7]  cluster [0,1,5,8]

    intervals   semitone offsets, e.g. [0, 4, 7] for a major chord.
    gain        mix level for non-root intervals (default 0.4).
    """
    intervals: list[int]
    gain:      float = 0.4


CHORD_PRESETS: dict[str, list[int]] = {
    "fifth":   [0, 7],
    "octave":  [0, 12],
    "minor":   [0, 3, 7],
    "major":   [0, 4, 7],
    "cluster": [0, 1, 5, 8],
}


# four-layer default stack for stretch ratios 200%-800%+.
# layers differ in grain size, stretch modifier, and pitch offset.

DEFAULT_LAYERS: list[LayerConfig] = [
    # foundation: full stretch, wide grains, no pitch offset
    LayerConfig(
        stretch_modifier=1.00, pitch_ratio=1.0,
        grain_ms=180, overlap=0.75, jitter_ms=15,
        n_avg=3, avg_spread=0.40, gain=1.00, phase_rand_ms=5.0,
    ),
    # shimmer: 85% stretch, narrow grains, +7 cents
    LayerConfig(
        stretch_modifier=0.85, pitch_ratio=cents_to_ratio(7),
        grain_ms=70,  overlap=0.82, jitter_ms=28,
        n_avg=2, avg_spread=0.65, gain=0.55, phase_rand_ms=12.0,
    ),
    # depth: 115% stretch, wide grains, -12 cents
    LayerConfig(
        stretch_modifier=1.15, pitch_ratio=cents_to_ratio(-12),
        grain_ms=220, overlap=0.70, jitter_ms=10,
        n_avg=4, avg_spread=0.30, gain=0.45, phase_rand_ms=6.0,
    ),
    # dust: micro-grains, high jitter, +3 cents
    LayerConfig(
        stretch_modifier=0.97, pitch_ratio=cents_to_ratio(3),
        grain_ms=35,  overlap=0.88, jitter_ms=40,
        n_avg=2, avg_spread=0.80, gain=0.25, phase_rand_ms=15.0,
    ),
]


# ---------------------------------------------------------------------------
#  core stretcher
# ---------------------------------------------------------------------------

class LaminarStretcher:
    """
    granular time-stretcher.

    stretch_percent = 100  ->  output is 2x the input duration
    stretch_percent = 800  ->  output is 9x the input duration

    each layer is stretched independently then mixed.

    auto_grain   scales grain_ms with the stretch factor:
                   grain_ms * stretch_factor^0.2
                 (1x -> x1.0, 3x -> x1.25, 9x -> x1.55)

    fade_ms      cosine fade-in and fade-out applied to the output.
                 removes clicks at grain boundaries. 30 ms default.
    """

    def __init__(
        self,
        stretch_percent: float,
        layers:     Optional[list[LayerConfig]] = None,
        auto_grain: bool = True,
        fade_ms:    float = 30.0,
        lfos:       Optional[list[LFO]] = None,
        chord:      Optional[ChordMode] = None,
        seed:       Optional[int] = None,
    ) -> None:
        if stretch_percent <= -100:
            raise ValueError("stretch_percent must be > -100  (output would be zero or negative length)")
        self.stretch_factor = 1.0 + stretch_percent / 100.0
        self.layers     = layers if layers is not None else DEFAULT_LAYERS
        self.auto_grain = auto_grain
        self.fade_ms    = fade_ms
        self.lfos       = lfos if lfos is not None else []
        self.chord      = chord
        self.rng        = np.random.default_rng(seed)

    # -- grain engine ----------------------------------------------------------

    @staticmethod
    def _hann(n: int) -> np.ndarray:
        return np.hanning(n).astype(np.float64)

    def _make_lfo_signal(self, lfo: LFO, n_samples: int, sample_rate: int) -> np.ndarray:
        """
        generates a normalised LFO waveform of length n_samples in [-1, 1].
        phase is keyed to output sample index so rate_hz is constant
        regardless of stretch factor.
        """
        t = np.arange(n_samples, dtype=np.float64) / sample_rate

        if lfo.waveform == "sine":
            return np.sin(2.0 * np.pi * lfo.rate_hz * t)

        elif lfo.waveform == "triangle":
            # 1 - 4|frac(f*t) - 0.5| gives a triangle starting at -1,
            # peaking at +1 at the half-period, returning to -1 each cycle.
            phase = (lfo.rate_hz * t) % 1.0
            return 1.0 - 4.0 * np.abs(phase - 0.5)

        elif lfo.waveform == "random_walk":
            # white noise LP-filtered to rate_hz via a first-order IIR.
            # alpha ~= 2*pi*fc/fs   (holds for fc << fs)
            # y[n] = alpha*x[n] + (1-alpha)*y[n-1]
            noise = self.rng.standard_normal(n_samples)
            alpha = float(np.clip(2.0 * np.pi * lfo.rate_hz / sample_rate, 1e-8, 1.0))
            sig   = lfilter([alpha], [1.0, -(1.0 - alpha)], noise)
            # normalise using 3-sigma so [-1, 1] is occupied on average
            std = float(np.std(sig))
            if std > 1e-8:
                sig /= 3.0 * std
            return np.clip(sig, -1.0, 1.0)

        else:
            return np.zeros(n_samples, dtype=np.float64)

    def _blend_grain(
        self,
        src: np.ndarray,
        center: int,
        grain_size: int,
        n_avg: int,
        avg_spread: float,
    ) -> np.ndarray:
        """
        returns a Hann-weighted average of n_avg source grains from a
        neighbourhood of radius (avg_spread * grain_size) around center.
        """
        n = len(src)
        center = int(np.clip(center, 0, n - grain_size))

        if n_avg <= 1 or avg_spread == 0.0:
            return src[center:center + grain_size].astype(np.float64)

        radius = int(grain_size * avg_spread)
        offsets = np.linspace(-radius, radius, n_avg, dtype=int)

        # centre-emphasised weights (Hann + small floor so edges always contribute)
        weights = np.hanning(n_avg) + 0.1
        weights /= weights.sum()

        result = np.zeros(grain_size, dtype=np.float64)
        for w, off in zip(weights, offsets):
            s = int(np.clip(center + int(off), 0, n - grain_size))
            result += w * src[s:s + grain_size].astype(np.float64)
        return result

    # -- single layer ----------------------------------------------------------

    def _stretch_layer(
        self,
        audio: np.ndarray,
        sample_rate: int,
        layer: LayerConfig,
    ) -> np.ndarray:
        """
        overlap-add granular stretch for one layer with optional LFO modulation.

        hop_out  output advance per grain  (grain_size * (1 - overlap))
        hop_in   source advance per grain  (hop_out * pitch_ratio / eff_stretch)

        when hop_in < hop_out the source is read more slowly than the output grows.
        pitch_ratio adjusts the scan speed independently of the stretch factor.
        jitter scatters each grain's source position by +- jitter_ms samples.

        LFOs targeting "grain_ms" and "jitter_ms" are applied per grain.
        LFOs targeting "gain" are applied as a post-loop multiply.
        """
        eff_stretch = self.stretch_factor * layer.stretch_modifier

        # scale grain size with the stretch factor
        grain_ms   = layer.grain_ms * (self.stretch_factor ** 0.2) if self.auto_grain else layer.grain_ms
        grain_size = max(64, int(sample_rate * grain_ms / 1000))
        if grain_size % 2:
            grain_size += 1  # keep even for clean windowing

        hop_out = max(1, int(grain_size * (1.0 - layer.overlap)))
        hop_in  = hop_out * layer.pitch_ratio / eff_stretch
        jitter  = max(0, int(sample_rate * layer.jitter_ms / 1000))

        n_in  = len(audio)
        n_out = int(round(n_in * eff_stretch))

        # pre-compute LFO modulation arrays
        # each array sums over all LFOs targeting that parameter.
        # applied as: effective_value = base_value * (1 + mod[out_pos])
        grain_ms_mod  = np.zeros(n_out, dtype=np.float64)
        jitter_ms_mod = np.zeros(n_out, dtype=np.float64)
        gain_mod      = np.zeros(n_out, dtype=np.float64)
        has_grain_lfo  = False
        has_jitter_lfo = False
        has_gain_lfo   = False

        for lfo in self.lfos:
            sig = self._make_lfo_signal(lfo, n_out, sample_rate)
            if lfo.target == "grain_ms":
                grain_ms_mod  += lfo.depth * sig
                has_grain_lfo  = True
            elif lfo.target == "jitter_ms":
                jitter_ms_mod += lfo.depth * sig
                has_jitter_lfo = True
            elif lfo.target == "gain":
                gain_mod      += lfo.depth * sig
                has_gain_lfo   = True

        # allocate buffers for the maximum possible grain size.
        # worst case: grain_ms grows by the sum of all grain LFO depths.
        max_grain_expansion = sum(lfo.depth for lfo in self.lfos if lfo.target == "grain_ms")
        max_gs = max(grain_size,
                     int(sample_rate * grain_ms * (1.0 + max_grain_expansion) / 1000) + 2)
        max_gs += max_gs % 2

        out_buf  = np.zeros(n_out + max_gs, dtype=np.float64)
        norm_buf = np.zeros(n_out + max_gs, dtype=np.float64)

        # cache Hann windows to avoid recomputing when LFO grain size changes slowly
        _win_cache: dict[int, np.ndarray] = {}

        def get_win(gs: int) -> np.ndarray:
            if gs not in _win_cache:
                _win_cache[gs] = self._hann(gs)
            return _win_cache[gs]

        in_pos  = 0.0
        out_pos = 0

        total_grains  = max(1, n_out // hop_out)
        grains_done   = 0
        report_every  = max(1, total_grains // 40)

        while out_pos < n_out:
            # effective grain size for this grain
            if has_grain_lfo:
                mod = float(grain_ms_mod[out_pos])
                gs  = max(64, int(sample_rate * grain_ms * max(0.1, 1.0 + mod) / 1000))
                if gs % 2:
                    gs += 1
            else:
                gs = grain_size

            win_g = get_win(gs)

            # effective jitter for this grain
            if has_jitter_lfo:
                mod   = float(jitter_ms_mod[out_pos])
                j_eff = max(0, int(sample_rate * layer.jitter_ms * max(0.0, 1.0 + mod) / 1000))
            else:
                j_eff = jitter

            # source position
            center = int(round(in_pos))
            if j_eff:
                center += int(self.rng.integers(-j_eff, j_eff + 1))

            # fine scatter applied after jitter, independent per layer,
            # to reduce inter-layer phase coherence
            if layer.phase_rand_ms > 0:
                pr = int(sample_rate * layer.phase_rand_ms / 1000)
                center += int(self.rng.integers(-pr, pr + 1))

            # blend and window the grain
            grain  = self._blend_grain(audio, center, gs, layer.n_avg, layer.avg_spread)
            grain *= win_g

            end = min(out_pos + gs, len(out_buf))
            gl  = end - out_pos
            out_buf [out_pos:end] += grain[:gl]
            norm_buf[out_pos:end] += win_g[:gl]

            in_pos  += hop_in
            out_pos += hop_out
            grains_done += 1

            if grains_done % report_every == 0:
                pct = min(100, int(100 * out_pos / n_out))
                print(f"\r      {pct:3d} %  ", end="", flush=True)

        print(f"\r      100 %   ", flush=True)

        # divide by accumulated Hann window to restore amplitude
        safe = norm_buf > 1e-8
        out_buf[safe] /= norm_buf[safe]
        result = out_buf[:n_out]

        # gain LFO: per-sample multiply applied after the loop
        if has_gain_lfo:
            result *= 1.0 + gain_mod

        return result

    # -- output polish ---------------------------------------------------------

    def _apply_fades(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        short cosine fade-in and fade-out. removes clicks at grain boundaries
        when the source starts or ends mid-cycle.
        """
        n = audio.shape[0]
        fade_len = min(int(sample_rate * self.fade_ms / 1000), n // 4)
        if fade_len < 2:
            return audio

        ramp = (1.0 - np.cos(np.linspace(0.0, np.pi, fade_len))) * 0.5
        out  = audio.copy()

        if out.ndim == 2:
            out[:fade_len]  *= ramp   [:, None]
            out[-fade_len:] *= ramp[::-1, None]
        else:
            out[:fade_len]  *= ramp
            out[-fade_len:] *= ramp[::-1]

        return out

    # -- public API ------------------------------------------------------------

    def _chord_layer_list(self) -> list[tuple[LayerConfig, float]]:
        """
        expands self.layers into (layer, gain_multiplier) pairs.

        without chord mode: one entry per layer, multiplier = 1.0.
        with chord mode: one copy of the layer stack per interval.
          interval 0 (root): layers unchanged, multiplier = 1.0.
          other intervals: pitch_ratio multiplied by 2^(semitones/12),
                           multiplier = chord.gain.
        """
        if self.chord is None:
            return [(layer, 1.0) for layer in self.layers]

        result: list[tuple[LayerConfig, float]] = []
        for interval in self.chord.intervals:
            pitch_mult = 2.0 ** (interval / 12.0)
            gain_mult  = 1.0 if interval == 0 else self.chord.gain
            for layer in self.layers:
                shifted = replace(layer, pitch_ratio=layer.pitch_ratio * pitch_mult)
                result.append((shifted, gain_mult))
        return result

    def process_channel(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """process one mono channel; returns float64 (not yet normalised)."""
        n_target   = int(round(len(audio) * self.stretch_factor))
        mixed      = np.zeros(n_target, dtype=np.float64)
        chord_list = self._chord_layer_list()

        for i, (layer, gain_mult) in enumerate(chord_list, 1):
            print(
                f"    [{i}/{len(chord_list)}]  "
                f"mod={layer.stretch_modifier:.2f}x  "
                f"grain={layer.grain_ms:.0f} ms  "
                f"pitch={layer.pitch_ratio:+.4f}  "
                f"gain={layer.gain * gain_mult:.2f}"
            )
            stretched = self._stretch_layer(audio, sample_rate, layer)

            # trim/pad to the same target length; slight differences arise
            # from float rounding of eff_stretch
            if len(stretched) > n_target:
                stretched = stretched[:n_target]
            elif len(stretched) < n_target:
                stretched = np.pad(stretched, (0, n_target - len(stretched)))

            mixed += stretched * layer.gain * gain_mult

        return mixed

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        process mono or stereo float64 audio. returns float64, peak-normalised to +-1.

        if both stereo channels are identical, processes once and copies the result.
        """
        # if both channels are identical, process once and copy the result
        duplicate_ch = (
            audio.ndim == 2
            and audio.shape[1] == 2
            and np.array_equal(audio[:, 0], audio[:, 1])
        )

        if duplicate_ch:
            channels = [audio[:, 0]]
            print("  (L and R identical -- processing once, result copied to both channels)")
        elif audio.ndim == 2:
            channels = [audio[:, c] for c in range(audio.shape[1])]
        else:
            channels = [audio]

        results = []
        for c, ch in enumerate(channels):
            print(f"  Channel {c + 1} / {len(channels)}")
            results.append(self.process_channel(ch.astype(np.float64), sample_rate))

        if duplicate_ch:
            results = [results[0], results[0]]

        out = np.stack(results, axis=1) if len(results) > 1 else results[0]
        out = self._apply_fades(out, sample_rate)

        peak = np.max(np.abs(out))
        if peak > 1e-8:
            out /= peak
        return out


# ---------------------------------------------------------------------------
#  reverb
# ---------------------------------------------------------------------------

class LaminarReverb:
    """
    convolution reverb using a procedurally generated impulse response.

    IR structure:
      early reflections  -- 14 sparse echoes in the first 80 ms
      diffuse tail       -- dense exponentially-decaying noise from ~30 ms.
                            a one-pole LP filter applies frequency-dependent
                            absorption (damping). stereo width is set by
                            blending shared and per-channel noise.

    output is always stereo and includes the reverb tail beyond the dry signal.
    peak-normalised after mixing wet + dry.

    parameters
    ----------
    room_size    0-1; controls decay time (0 -> 0.5 s, 1 -> 8 s).
    damping      0-1; high-frequency absorption in the tail.
    wet          level of the reverb signal in the mix (0-1).
    dry          level of the dry signal in the mix (0-1).
    width        stereo spread of the reverb tail (0-1).
    pre_delay_ms silence before the first reflection, in ms.
    seed         RNG seed for the IR noise.
    """

    def __init__(
        self,
        room_size:    float = 0.65,
        damping:      float = 0.35,
        wet:          float = 0.25,
        dry:          float = 1.0,
        width:        float = 0.80,
        pre_delay_ms: float = 12.0,
        seed:         int   = 1,
    ) -> None:
        self.room_size    = float(np.clip(room_size, 0.0, 1.0))
        self.damping      = float(np.clip(damping,   0.0, 0.99))
        self.wet          = wet
        self.dry          = dry
        self.width        = float(np.clip(width, 0.0, 1.0))
        self.pre_delay_ms = pre_delay_ms
        self.seed         = seed

    def decay_seconds(self) -> float:
        """reverb decay time in seconds (-60 dB point)."""
        return 0.5 + self.room_size ** 1.5 * 7.5

    def _make_ir(self, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        """
        builds left and right impulse responses.

        0 -> pre_delay           : silence
        pre_delay -> +80 ms      : sparse early reflections
        pre_delay + 30 ms -> end : dense diffuse tail
        """
        rng     = np.random.default_rng(self.seed)
        decay_s = self.decay_seconds()
        pre_s   = self.pre_delay_ms / 1000.0
        ir_len  = int(sample_rate * (decay_s + pre_s + 0.05))

        ir_l = np.zeros(ir_len)
        ir_r = np.zeros(ir_len)

        # early reflections
        er_end_s = pre_s + 0.080
        er_times = np.sort(rng.uniform(pre_s, er_end_s, 14))
        for t_er in er_times:
            idx = int(t_er * sample_rate)
            if idx >= ir_len:
                continue
            amp = np.exp(-t_er * 6.0)
            ir_l[idx] += amp * (1.0 + rng.normal(0, 0.08))
            ir_r[idx] += amp * (1.0 + rng.normal(0, 0.08))

        # diffuse tail
        tail_start = int(sample_rate * (pre_s + 0.030))
        tail_len   = ir_len - tail_start
        if tail_len <= 0:
            return ir_l, ir_r

        t_tail   = np.arange(tail_len) / sample_rate
        tau      = decay_s / np.log(1000.0)        # -60 dB at decay_s
        envelope = np.exp(-t_tail / tau)

        common   = rng.normal(0, 1, tail_len)
        unique_l = rng.normal(0, 1, tail_len)
        unique_r = rng.normal(0, 1, tail_len)

        # equal-power blend of shared and per-channel noise
        w = self.width
        a = np.sqrt(max(0.0, 1.0 - w * 0.5))
        b = np.sqrt(w * 0.5)

        tail_l = a * common + b * unique_l
        tail_r = a * common + b * unique_r

        # one-pole LP filter for frequency-dependent absorption
        if self.damping > 0:
            d = self.damping * 0.92
            tail_l = lfilter([1.0 - d], [1.0, -d], tail_l)
            tail_r = lfilter([1.0 - d], [1.0, -d], tail_r)

        ir_l[tail_start:] += envelope * tail_l
        ir_r[tail_start:] += envelope * tail_r

        peak = max(np.max(np.abs(ir_l)), np.max(np.abs(ir_r)), 1e-8)
        ir_l /= peak
        ir_r /= peak

        return ir_l, ir_r

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        applies reverb. always returns stereo float64, peak-normalised.
        output is longer than the input by the reverb tail duration.
        """
        decay_s = self.decay_seconds()
        print(f"  Building IR  room={self.room_size:.2f}  decay={decay_s:.1f}s  "
              f"damp={self.damping:.2f}  width={self.width:.2f}")

        ir_l, ir_r = self._make_ir(sample_rate)

        if audio.ndim == 2:
            send = np.mean(audio, axis=1)
            dry_l, dry_r = audio[:, 0], audio[:, 1]
        else:
            send = audio
            dry_l = dry_r = audio

        print("  Convolving...")
        wet_l = oaconvolve(send, ir_l)
        wet_r = oaconvolve(send, ir_r)

        out_len = len(wet_l)
        dry_l = np.pad(dry_l.astype(np.float64), (0, out_len - len(dry_l)))
        dry_r = np.pad(dry_r.astype(np.float64), (0, out_len - len(dry_r)))

        out_l = self.dry * dry_l + self.wet * wet_l
        out_r = self.dry * dry_r + self.wet * wet_r

        out  = np.stack([out_l, out_r], axis=1)
        peak = np.max(np.abs(out))
        if peak > 1e-8:
            out /= peak
        return out


# ---------------------------------------------------------------------------
#  compressor
# ---------------------------------------------------------------------------

class LaminarCompressor:
    """
    soft-knee RMS compressor.

    1. compute per-sample RMS envelope with a sliding window (uniform_filter1d).
    2. map through a soft-knee static curve to get gain reduction in dB.
    3. attack: minimum_filter1d over a short window holds maximum reduction
       so gain clamps quickly when a loud passage arrives.
    4. release: one-pole IIR (lfilter) gives exponential gain recovery after
       the loud passage ends. O(N) regardless of release time.
    5. apply gain (+ makeup) to all channels using the same curve (stereo-linked).

    parameters
    ----------
    threshold_db   level above which gain reduction begins (dBFS, default -20).
    ratio          compression ratio above threshold (default 3.0 = 3:1).
    knee_db        soft-knee transition width (default 6 dB).
    makeup_db      gain added after compression (default 3 dB).
    attack_ms      gain reduction window length (default 30 ms).
    release_ms     gain recovery time (default 200 ms).
    rms_window_ms  sliding-window length for RMS detection (default 100 ms).
    """

    def __init__(
        self,
        threshold_db:  float = -20.0,
        ratio:         float = 3.0,
        knee_db:       float = 6.0,
        makeup_db:     float = 3.0,
        attack_ms:     float = 30.0,
        release_ms:    float = 200.0,
        rms_window_ms: float = 100.0,
    ) -> None:
        self.threshold_db  = threshold_db
        self.ratio         = max(1.0, ratio)
        self.knee_db       = max(0.0, knee_db)
        self.makeup_db     = makeup_db
        self.attack_ms     = attack_ms
        self.release_ms    = release_ms
        self.rms_window_ms = rms_window_ms

    def _rms_envelope(self, mono: np.ndarray, sample_rate: int) -> np.ndarray:
        win = max(3, int(sample_rate * self.rms_window_ms / 1000))
        return np.sqrt(uniform_filter1d(mono ** 2, size=win, mode='mirror'))

    def _gain_reduction_db(self, rms: np.ndarray) -> np.ndarray:
        """soft-knee static compression curve; returns gain reduction in dB (<= 0)."""
        db = 20.0 * np.log10(np.maximum(rms, 1e-9))
        T, R, W = self.threshold_db, self.ratio, self.knee_db

        x = db - T                          # distance above threshold

        below = db                          # no reduction
        above = T + x / R                   # full compression
        knee  = db + (1.0 / R - 1.0) * (x + W / 2.0) ** 2 / (2.0 * W)

        out_db = np.where(
            2.0 * x < -W, below,
            np.where(2.0 * x > W, above, knee),
        )
        return out_db - db                  # always <= 0

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """apply compression; returns float64 (not peak-normalised)."""
        mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio

        rms     = self._rms_envelope(mono, sample_rate)
        gain_db = self._gain_reduction_db(rms)   # <= 0

        # attack: minimum_filter holds maximum reduction in a short window,
        # so gain clamps quickly when a loud passage arrives.
        # bidirectional window gives a small look-ahead.
        attack_samps = max(3, int(self.attack_ms * sample_rate / 1000))
        gain_db = minimum_filter1d(gain_db, size=attack_samps * 2 + 1)

        # release: one-pole IIR gives exponential recovery, O(N) regardless of
        # time constant. alpha = 1 - exp(-1/tau) where tau = release_ms in samples.
        tau   = max(1.0, self.release_ms * sample_rate / 1000.0)
        alpha = float(1.0 - np.exp(-1.0 / tau))
        gain_db = lfilter([alpha], [1.0, -(1.0 - alpha)], gain_db)

        gain = 10.0 ** ((gain_db + self.makeup_db) / 20.0)

        out = audio * gain[:, None] if audio.ndim == 2 else audio * gain

        # report dynamic range change
        dr_in  = 20 * np.log10(max(np.max(np.abs(audio)), 1e-9) /
                                max(np.sqrt(np.mean(audio ** 2)), 1e-9))
        dr_out = 20 * np.log10(max(np.max(np.abs(out)),   1e-9) /
                                max(np.sqrt(np.mean(out ** 2)),   1e-9))
        print(f"  Peak-to-RMS before: {dr_in:.1f} dB  ->  after: {dr_out:.1f} dB")

        return out


# ---------------------------------------------------------------------------
#  tone correction
# ---------------------------------------------------------------------------

class ToneCorrector:
    """
    two biquad filters applied in series.

    1. low shelf  -- boosts below shelf_hz. Audio EQ Cookbook bilinear-transform,
                     shelf slope S=1.
    2. high cut   -- second-order Butterworth low-pass at highcut_hz.

    coefficients are computed once at construction.
    applied per-channel via lfilter; works on mono or stereo.

    parameters
    ----------
    sample_rate   sample rate in Hz.
    shelf_hz      low-shelf corner frequency (default 220 Hz).
    shelf_db      low-shelf boost in dB (default +3.5 dB).
    highcut_hz    high-cut -3 dB point (default 7500 Hz).
    """

    def __init__(
        self,
        sample_rate: int,
        shelf_hz:    float = 220.0,
        shelf_db:    float = 3.5,
        highcut_hz:  float = 7500.0,
    ) -> None:
        self.sample_rate = sample_rate
        self._b_shelf, self._a_shelf = self._low_shelf(shelf_hz, shelf_db)
        self._b_high,  self._a_high  = self._highcut(highcut_hz)

    def _low_shelf(
        self, shelf_hz: float, gain_db: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Audio EQ Cookbook low-shelf biquad, shelf slope S=1.

        A     = 10^(dBgain/40)
        w0    = 2*pi*f0/fs
        alpha = sin(w0)/sqrt(2)

        coefficients divided by a0:
          b = [A*((A+1)-(A-1)*cos+2*sqA*alpha), 2A*((A-1)-(A+1)*cos), A*((A+1)-(A-1)*cos-2*sqA*alpha)]
          a = [  (A+1)+(A-1)*cos+2*sqA*alpha, -2*((A-1)+(A+1)*cos),   (A+1)+(A-1)*cos-2*sqA*alpha  ]
        """
        A     = 10.0 ** (gain_db / 40.0)
        w0    = 2.0 * np.pi * shelf_hz / self.sample_rate
        cos_w = np.cos(w0)
        sin_w = np.sin(w0)
        alpha = sin_w / np.sqrt(2.0)
        sqA   = np.sqrt(A)

        b0 =    A * ((A + 1) - (A - 1) * cos_w + 2 * sqA * alpha)
        b1 =  2*A * ((A - 1) - (A + 1) * cos_w)
        b2 =    A * ((A + 1) - (A - 1) * cos_w - 2 * sqA * alpha)
        a0 =         (A + 1) + (A - 1) * cos_w + 2 * sqA * alpha
        a1 =   -2 * ((A - 1) + (A + 1) * cos_w)
        a2 =         (A + 1) + (A - 1) * cos_w - 2 * sqA * alpha

        return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])

    def _highcut(self, highcut_hz: float) -> tuple[np.ndarray, np.ndarray]:
        """second-order Butterworth low-pass."""
        nyq = self.sample_rate / 2.0
        wn  = min(highcut_hz / nyq, 0.9999)   # clamp safely below Nyquist
        return butter(2, wn, btype='low')

    def process(self, audio: np.ndarray) -> np.ndarray:
        """apply shelf then high-cut per channel; returns same shape."""
        def apply(ch: np.ndarray) -> np.ndarray:
            x = lfilter(self._b_shelf, self._a_shelf, ch)
            return lfilter(self._b_high,  self._a_high,  x)

        if audio.ndim == 2:
            return np.stack(
                [apply(audio[:, c]) for c in range(audio.shape[1])], axis=1
            )
        return apply(audio)


# ---------------------------------------------------------------------------
#  I/O helpers
# ---------------------------------------------------------------------------

def load_wav(path: str) -> tuple[int, np.ndarray]:
    """read a .wav file and normalise to float64 in [-1, 1]."""
    rate, data = wavfile.read(path)
    if   data.dtype == np.int16:    return rate, data.astype(np.float64) / 32_768.0
    elif data.dtype == np.int32:    return rate, data.astype(np.float64) / 2_147_483_648.0
    elif data.dtype == np.float32:  return rate, data.astype(np.float64)
    return rate, data.astype(np.float64)


def save_wav(path: str, rate: int, data: np.ndarray) -> None:
    """write float64 audio to a 16-bit .wav file."""
    out = (np.clip(data, -1.0, 1.0) * 32_767).astype(np.int16)
    wavfile.write(path, rate, out)


# ---------------------------------------------------------------------------
#  pitch analysis and MIDI export
# ---------------------------------------------------------------------------

def _var_len(n: int) -> bytes:
    """encode a non-negative integer as a MIDI variable-length quantity."""
    data = [n & 0x7F]
    n >>= 7
    while n:
        data.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(data))


def _merge_midi_notes(
    notes: list[tuple[float, float, int, int]],
    gap_s: float,
) -> list[tuple[float, float, int, int]]:
    """
    merge runs of same-pitch notes where the gap between consecutive notes
    is <= gap_s seconds. the merged note spans from the first start to the
    last end; velocity is the max of the components.
    """
    if not notes:
        return notes

    by_pitch: dict[int, list[tuple[float, float, int]]] = {}
    for start_s, dur_s, note, vel in notes:
        by_pitch.setdefault(note, []).append((start_s, dur_s, vel))

    merged: list[tuple[float, float, int, int]] = []
    for note, segs in by_pitch.items():
        segs.sort()
        cur_start, cur_dur, cur_vel = segs[0]
        cur_end = cur_start + cur_dur
        for start_s, dur_s, vel in segs[1:]:
            end_s = start_s + dur_s
            if start_s - cur_end <= gap_s:
                cur_end = max(cur_end, end_s)
                cur_vel = max(cur_vel, vel)
            else:
                merged.append((cur_start, cur_end - cur_start, note, cur_vel))
                cur_start, cur_end, cur_vel = start_s, end_s, vel
        merged.append((cur_start, cur_end - cur_start, note, cur_vel))

    merged.sort(key=lambda x: x[0])
    return merged


def _write_midi_file(
    notes: list[tuple[float, float, int, int]],
    path:  str,
    bpm:   float = 120.0,
    ppqn:  int   = 960,
) -> None:
    """
    write a type-0 MIDI file.
    notes = [(start_s, dur_s, midi_note, velocity), ...]

    bpm must match the DAW project tempo. at 120 BPM with ppqn=960,
    there are 1920 ticks per second, so a note at t=1.0s lands on beat 3.
    use --midi-bpm to match your project if it differs from 120.
    """
    uspb          = int(60_000_000 / bpm)    # microseconds per beat
    ticks_per_sec = ppqn * bpm / 60.0

    def to_tick(t_s: float) -> int:
        return int(round(t_s * ticks_per_sec))

    events: list[tuple[int, bytes]] = []

    # tempo
    events.append((0, bytes([0xFF, 0x51, 0x03,
                              (uspb >> 16) & 0xFF,
                              (uspb >> 8)  & 0xFF,
                               uspb        & 0xFF])))
    # track name
    name = b'Laminar Pitch'
    events.append((0, bytes([0xFF, 0x03]) + _var_len(len(name)) + name))

    for start_s, dur_s, note, vel in notes:
        on_t  = to_tick(start_s)
        off_t = max(to_tick(start_s + dur_s), on_t + 1)
        events.append((on_t,  bytes([0x90, note & 0x7F, vel & 0x7F])))
        events.append((off_t, bytes([0x80, note & 0x7F, 0x00])))

    # note-off before note-on at the same tick, avoids hanging notes
    events.sort(key=lambda e: (e[0], 0 if (e[1][0] & 0xF0) == 0x80 else 1))

    track_data = b''
    prev = 0
    for tick, msg in events:
        track_data += _var_len(tick - prev) + msg
        prev = tick
    track_data += b'\x00\xFF\x2F\x00'  # end of track

    header = (b'MThd'
              + (6).to_bytes(4, 'big')
              + (0).to_bytes(2, 'big')   # format 0
              + (1).to_bytes(2, 'big')   # one track
              + ppqn.to_bytes(2, 'big'))
    track  = b'MTrk' + len(track_data).to_bytes(4, 'big') + track_data

    with open(path, 'wb') as f:
        f.write(header + track)


class PitchAnalyzer:
    """
    monophonic pitch detector and MIDI exporter.

    uses YIN to estimate the fundamental frequency of each analysis frame,
    then segments the results into discrete notes.

    the MIDI output uses SMPTE timecode (25 fps, 40 ticks/frame = 1000 ticks/sec).
    timing is absolute and independent of DAW project tempo.

    parameters
    ----------
    fmin           lowest detectable pitch in Hz (default 50 Hz).
    fmax           highest detectable pitch in Hz (default 1000 Hz).
    frame_ms       analysis window length in ms (default 100 ms).
    hop_ms         hop between windows in ms (default 50 ms).
    confidence     frames below this threshold are treated as unvoiced (default 0.6).
    min_note_s     minimum note duration in seconds; shorter segments are dropped.
    semitone_tol   pitch drift in semitones before starting a new note (default 0.8).
    """

    _YIN_THRESHOLD = 0.15   # internal CMNDF threshold; standard YIN default

    def __init__(
        self,
        fmin:         float = 50.0,
        fmax:         float = 1000.0,
        frame_ms:     float = 100.0,
        hop_ms:       float = 50.0,
        confidence:   float = 0.60,
        min_note_s:   float = 0.15,
        semitone_tol: float = 0.8,
    ) -> None:
        self.fmin         = fmin
        self.fmax         = fmax
        self.frame_ms     = frame_ms
        self.hop_ms       = hop_ms
        self.confidence   = confidence
        self.min_note_s   = min_note_s
        self.semitone_tol = semitone_tol

    # -- YIN internals ---------------------------------------------------------

    @staticmethod
    def _yin_diff(frame: np.ndarray) -> np.ndarray:
        """
        YIN difference function d[t] = sum (x[n] - x[n+t])^2

        expanded as d[t] = sum x[n]^2 + sum x[n+t]^2 - 2*ACF[t]
        computed via FFT autocorrelation.
        """
        N    = len(frame)
        half = N // 2

        n_fft = 1 << (2 * N - 1).bit_length()   # next power-of-2 >= 2N-1
        F   = np.fft.rfft(frame, n=n_fft)
        acf = np.fft.irfft(F * F.conj())[:half + 1].real

        cs   = np.concatenate(([0.0], np.cumsum(frame ** 2)))
        taus = np.arange(half + 1)
        p1   = cs[N - taus]          # sum x[n]^2    for n = 0 .. N-t-1
        p2   = cs[N] - cs[taus]      # sum x[n+t]^2  for same range

        d    = p1 + p2 - 2.0 * acf
        d[0] = 0.0
        return d

    @staticmethod
    def _cmndf(d: np.ndarray) -> np.ndarray:
        """cumulative mean normalised difference function (CMNDF)."""
        cmndf   = np.ones_like(d)
        cumsum  = np.cumsum(d)
        running = cumsum - d[0]                      # sum d[1..t]
        taus    = np.arange(len(d), dtype=np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            cmndf[1:] = np.where(
                running[1:] > 0,
                d[1:] * taus[1:] / running[1:],
                1.0,
            )
        cmndf[0] = 1.0
        return cmndf

    def _estimate_pitch(
        self, frame: np.ndarray, sample_rate: int
    ) -> tuple[float, float]:
        """
        YIN pitch estimate for one frame. returns (freq_hz, confidence).
        confidence = 1 - CMNDF[tau_best].
        returns (0, 0) for silent or unvoiced frames.
        """
        if np.sqrt(np.mean(frame ** 2)) < 1e-5:
            return 0.0, 0.0

        N       = len(frame)
        min_lag = max(1, int(sample_rate / self.fmax))
        max_lag = min(N // 2 - 1, int(sample_rate / self.fmin))
        if max_lag <= min_lag:
            return 0.0, 0.0

        d      = self._yin_diff(frame)
        cmnd   = self._cmndf(d)
        region = cmnd[min_lag: max_lag + 1]

        # find first dip below the YIN threshold
        below = np.where(region < self._YIN_THRESHOLD)[0]
        if len(below) == 0:
            # no clear pitch, fall back to global minimum
            tau_local = int(np.argmin(region))
            if region[tau_local] > 0.45:
                return 0.0, 0.0
        else:
            # slide to the bottom of the dip
            tau_local = int(below[0])
            while (tau_local + 1 < len(region) - 1
                   and region[tau_local + 1] < region[tau_local]):
                tau_local += 1

        tau = min_lag + tau_local

        # parabolic interpolation for sub-sample accuracy
        if 0 < tau < len(cmnd) - 1:
            s0, s1, s2 = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
            denom = 2.0 * s1 - s0 - s2
            tau_f = tau + (s0 - s2) / (2.0 * denom) if abs(denom) > 1e-8 else float(tau)
        else:
            tau_f = float(tau)

        freq = sample_rate / tau_f
        conf = max(0.0, 1.0 - cmnd[tau])
        return freq, conf

    # -- analysis --------------------------------------------------------------

    @staticmethod
    def _freq_to_midi(freq: float) -> float:
        """frequency in Hz to fractional MIDI note number."""
        if freq <= 0:
            return -1.0
        return 69.0 + 12.0 * np.log2(freq / 440.0)

    def analyze(
        self, audio: np.ndarray, sample_rate: int
    ) -> list[tuple[float, float, int, int]]:
        """
        detect pitches and return a list of MIDI notes.
        returns [(start_s, duration_s, midi_note, velocity), ...] sorted by start time.
        velocity (1-127) is derived from frame RMS.
        """
        mono = (np.mean(audio, axis=1) if audio.ndim == 2 else audio).astype(np.float64)

        frame_len = max(256, int(sample_rate * self.frame_ms / 1000))
        hop_len   = max(1,   int(sample_rate * self.hop_ms   / 1000))
        n         = len(mono)

        # pad so every frame is full-length
        pad    = frame_len - ((n - frame_len) % hop_len or hop_len)
        padded = np.pad(mono, (0, max(0, pad)))

        # frame-level pitch estimates
        frames: list[tuple[float, float, float, float]] = []  # (t, midi_f, conf, rms)
        n_frames = max(1, (n - frame_len) // hop_len + 1)
        report_every = max(1, n_frames // 20)

        for i, start in enumerate(range(0, n, hop_len)):
            frame  = padded[start: start + frame_len]
            t      = (start + frame_len / 2) / sample_rate
            freq, conf = self._estimate_pitch(frame, sample_rate)
            midi_f = self._freq_to_midi(freq)
            rms    = float(np.sqrt(np.mean(frame ** 2)))
            frames.append((t, midi_f, conf, rms))

            if i % report_every == 0:
                pct = min(100, int(100 * start / n))
                print(f"\r    {pct:3d} %  ", end="", flush=True)

        print(f"\r    100 %   ", flush=True)

        # note segmentation
        notes: list[tuple[float, float, int, int]] = []
        seg_start: Optional[float] = None
        seg_midis: list[float]     = []
        seg_rms:   list[float]     = []

        half_frame_s = (frame_len / 2) / sample_rate

        def flush(end_t: float) -> None:
            nonlocal seg_start, seg_midis, seg_rms
            if seg_start is None or not seg_midis:
                return
            dur = end_t - seg_start
            if dur >= self.min_note_s:
                note = int(np.clip(round(float(np.median(seg_midis))), 0, 127))
                vel  = int(np.clip(20 + float(np.median(seg_rms)) * 700, 20, 110))
                notes.append((seg_start, dur, note, vel))
            seg_start = None
            seg_midis.clear()
            seg_rms.clear()

        for t, midi_f, conf, rms in frames:
            voiced = conf >= self.confidence and 0.0 <= midi_f <= 127.0

            if not voiced:
                flush(t - half_frame_s)
                continue

            if seg_start is None:
                # begin new segment
                seg_start = t - half_frame_s
                seg_midis = [midi_f]
                seg_rms   = [rms]
            elif abs(midi_f - float(np.median(seg_midis))) > self.semitone_tol:
                # pitch drifted past tolerance, start a new segment
                flush(t - half_frame_s)
                seg_start = t - half_frame_s
                seg_midis = [midi_f]
                seg_rms   = [rms]
            else:
                seg_midis.append(midi_f)
                seg_rms.append(rms)

        if frames:
            flush(frames[-1][0] + half_frame_s)

        return sorted(notes)

    def write_midi(
        self,
        notes: list[tuple[float, float, int, int]],
        path:  str,
        bpm:   float = 120.0,
        ppqn:  int   = 960,
    ) -> None:
        """write notes to a type-0 MIDI file. bpm must match the DAW project tempo."""
        _write_midi_file(notes, path, bpm=bpm, ppqn=ppqn)


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="laminar_stretch",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input",  help="input .wav file")
    p.add_argument("output", help="output .wav file")

    g = p.add_argument_group("stretch")
    g.add_argument(
        "-s", "--stretch", type=float, default=200.0, metavar="PCT",
        help="stretch %%: 100 = 2x length, 800 = 9x length  (default: 200)",
    )
    g.add_argument(
        "-l", "--num-layers", type=int, default=4, metavar="N",
        help="number of layers 1-4  (default: 4)",
    )
    g.add_argument(
        "--grain-ms", type=float, default=None, metavar="MS",
        help="override base grain size in ms for all layers",
    )
    g.add_argument(
        "--no-auto-grain", action="store_true",
        help="disable automatic grain scaling at extreme stretch ratios",
    )
    g.add_argument(
        "--phase-rand", type=float, default=None, metavar="MS",
        help="set phase_rand_ms on all layers (ms)",
    )
    g.add_argument(
        "--no-phase-rand", action="store_true",
        help="disable phase randomisation on all layers",
    )
    g.add_argument(
        "--warmth", action="store_true",
        help=(
            "preset: grain-ms 280, tone correction (220 Hz shelf +3.5 dB, "
            "7500 Hz high-cut), phase-rand 10 ms.  explicit flags override any of these."
        ),
    )
    g.add_argument(
        "--fade-ms", type=float, default=30.0, metavar="MS",
        help="cosine fade-in/out length in ms  (default: 30)",
    )
    g.add_argument(
        "--seed", type=int, default=None,
        help="random seed",
    )
    g.add_argument(
        "--no-normalize", action="store_true",
        help="skip peak normalisation on the output",
    )

    r = p.add_argument_group("reverb  (post-processing)")
    r.add_argument(
        "--reverb", action="store_true",
        help="enable reverb",
    )
    r.add_argument(
        "--reverb-room", type=float, default=0.65, metavar="0-1",
        help="controls decay time 0.5 s -> 8 s  (default: 0.65)",
    )
    r.add_argument(
        "--reverb-damp", type=float, default=0.35, metavar="0-1",
        help="high-frequency absorption; 0=bright, 1=dark  (default: 0.35)",
    )
    r.add_argument(
        "--reverb-wet", type=float, default=0.25, metavar="0-1",
        help="wet signal level  (default: 0.25)",
    )
    r.add_argument(
        "--reverb-width", type=float, default=0.80, metavar="0-1",
        help="stereo spread of the reverb tail  (default: 0.80)",
    )
    r.add_argument(
        "--reverb-predelay", type=float, default=12.0, metavar="MS",
        help="silence before first reflection  (default: 12 ms)",
    )

    c = p.add_argument_group("compress  (post-processing)")
    c.add_argument(
        "--compress", action="store_true",
        help="enable dynamic range compression",
    )
    c.add_argument(
        "--compress-threshold", type=float, default=-20.0, metavar="dBFS",
        help="level above which gain reduction begins  (default: -20 dBFS)",
    )
    c.add_argument(
        "--compress-ratio", type=float, default=3.0, metavar="N",
        help="compression ratio above threshold  (default: 3.0)",
    )
    c.add_argument(
        "--compress-makeup", type=float, default=3.0, metavar="dB",
        help="makeup gain added after compression  (default: 3 dB)",
    )
    c.add_argument(
        "--compress-release", type=float, default=200.0, metavar="MS",
        help="gain recovery time after a loud passage  (default: 200 ms)",
    )

    t = p.add_argument_group("tone correction  (post-processing, after compression)")
    t.add_argument(
        "--tone", action="store_true",
        help="enable tone correction: low-shelf boost + Butterworth high cut",
    )
    t.add_argument(
        "--tone-shelf-hz", type=float, default=None, metavar="HZ",
        help="low-shelf corner frequency in Hz  (default: 220)",
    )
    t.add_argument(
        "--tone-shelf-db", type=float, default=None, metavar="dB",
        help="low-shelf boost in dB  (default: +3.5)",
    )
    t.add_argument(
        "--tone-highcut-hz", type=float, default=None, metavar="HZ",
        help="high-cut -3 dB point in Hz  (default: 7500)",
    )

    ch = p.add_argument_group("chord  (harmonic doubling)")
    ch.add_argument(
        "--chord",
        choices=list(CHORD_PRESETS),
        default=None, metavar="PRESET",
        help=(
            "copy the layer stack per interval and mix as a chord.  presets: "
            + "  ".join(f"{k} {v}" for k, v in CHORD_PRESETS.items())
        ),
    )
    ch.add_argument(
        "--chord-gain", type=float, default=0.4, metavar="0-1",
        help="mix level for non-root chord intervals  (default: 0.4)",
    )

    lfo = p.add_argument_group("lfo  (modulation)")
    lfo.add_argument(
        "--lfo", metavar="TARGET:WAVEFORM:RATE_HZ:DEPTH",
        action="append", default=[],
        help=(
            "add an LFO modulator.  repeat for multiple LFOs.  "
            "TARGET: grain_ms | jitter_ms | gain.  "
            "WAVEFORM: sine | triangle | random_walk.  "
            "example: --lfo grain_ms:sine:0.08:0.3"
        ),
    )

    m = p.add_argument_group("midi  (pitch analysis, post-processing)")
    m.add_argument(
        "--midi", default=None, metavar="PATH",
        help="analyse the output audio, detect pitches, write a MIDI file here",
    )
    m.add_argument(
        "--midi-fmin", type=float, default=50.0, metavar="HZ",
        help="lowest detectable pitch in Hz  (default: 50)",
    )
    m.add_argument(
        "--midi-fmax", type=float, default=1000.0, metavar="HZ",
        help="highest detectable pitch in Hz  (default: 1000)",
    )
    m.add_argument(
        "--midi-confidence", type=float, default=0.60, metavar="0-1",
        help="confidence threshold; lower = accept more frames  (default: 0.60)",
    )
    m.add_argument(
        "--midi-min-note", type=float, default=0.15, metavar="S",
        help="minimum note duration in seconds  (default: 0.15)",
    )
    m.add_argument(
        "--midi-frame", type=float, default=100.0, metavar="MS",
        help="analysis frame length in ms  (default: 100)",
    )
    m.add_argument(
        "--midi-hop", type=float, default=50.0, metavar="MS",
        help="hop between analysis frames in ms  (default: 50)",
    )
    m.add_argument(
        "--midi-bpm", type=float, default=120.0, metavar="BPM",
        help="DAW project tempo in BPM; MIDI ticks are scaled to match  (default: 120)",
    )
    m.add_argument(
        "--midi-merge", action="store_true",
        help="merge repeated same-pitch notes into a single long note",
    )
    m.add_argument(
        "--midi-merge-gap", type=float, default=500.0, metavar="MS",
        help="max gap in ms between notes to be merged  (default: 500)",
    )

    return p


def main() -> None:
    args = _build_parser().parse_args()

    num_layers = max(1, min(4, args.num_layers))
    layers = [replace(lc) for lc in DEFAULT_LAYERS[:num_layers]]

    # --warmth: set defaults before explicit overrides.
    # explicit flags win because we only set fields still None.
    if args.warmth:
        if args.grain_ms is None:
            args.grain_ms = 280.0
        args.tone = True            # tone_shelf_*/highcut_hz left None, picks up ToneCorrector defaults
        # phase_rand priority handled below

    # grain_ms override
    if args.grain_ms is not None:
        for lc in layers:
            lc.grain_ms = float(args.grain_ms)

    # phase randomisation priority order:
    #   --no-phase-rand > --phase-rand > --warmth (10 ms) > per-layer defaults
    if args.no_phase_rand:
        for lc in layers:
            lc.phase_rand_ms = 0.0
    elif args.phase_rand is not None:
        for lc in layers:
            lc.phase_rand_ms = float(args.phase_rand)
    elif args.warmth:
        for lc in layers:
            lc.phase_rand_ms = 10.0
    # else: per-layer defaults from DEFAULT_LAYERS

    # parse --lfo strings
    lfos: list[LFO] = []
    valid_targets   = {"grain_ms", "jitter_ms", "gain"}
    valid_waveforms = {"sine", "triangle", "random_walk"}
    for raw in args.lfo:
        parts = raw.split(":")
        if len(parts) != 4:
            raise SystemExit(
                f"error: --lfo requires exactly 4 colon-separated fields "
                f"(target:waveform:rate_hz:depth), got {raw!r}"
            )
        target, waveform, rate_str, depth_str = parts
        if target not in valid_targets:
            raise SystemExit(f"error: unknown LFO target {target!r}  (choose from {sorted(valid_targets)})")
        if waveform not in valid_waveforms:
            raise SystemExit(f"error: unknown LFO waveform {waveform!r}  (choose from {sorted(valid_waveforms)})")
        try:
            rate_hz = float(rate_str)
            depth   = float(depth_str)
        except ValueError:
            raise SystemExit(f"error: rate_hz and depth must be numbers in --lfo {raw!r}")
        lfos.append(LFO(rate_hz=rate_hz, depth=depth, target=target, waveform=waveform))

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"error: file not found: {in_path}")

    print("=" * 58)
    print("  Laminar Stretch")
    print("=" * 58)
    print(f"  input    : {in_path}")
    print(f"  output   : {args.output}")
    print(f"  stretch  : {args.stretch} %  ->  {1 + args.stretch / 100:.3f}x length")
    print(f"  layers   : {num_layers}")
    print(f"  auto_grain: {'off' if args.no_auto_grain else 'on'}")
    if args.reverb:
        reverb = LaminarReverb(
            room_size    = args.reverb_room,
            damping      = args.reverb_damp,
            wet          = args.reverb_wet,
            width        = args.reverb_width,
            pre_delay_ms = args.reverb_predelay,
        )
        decay_s = reverb.decay_seconds()
        print(f"  reverb   : room={args.reverb_room:.2f}  decay={decay_s:.1f}s  "
              f"damp={args.reverb_damp:.2f}  wet={args.reverb_wet:.2f}  width={args.reverb_width:.2f}")
    chord: Optional[ChordMode] = None
    if args.chord:
        chord = ChordMode(intervals=CHORD_PRESETS[args.chord], gain=args.chord_gain)
        print(f"  chord    : {args.chord}  intervals={chord.intervals}  gain={chord.gain:.2f}")
    for lfo in lfos:
        print(f"  lfo      : {lfo.target}  {lfo.waveform}  {lfo.rate_hz:.3f} Hz  depth {lfo.depth:.2f}")
    if args.compress:
        print(f"  compress : threshold={args.compress_threshold:.0f} dBFS  "
              f"ratio={args.compress_ratio:.1f}:1  "
              f"makeup={args.compress_makeup:+.0f} dB  "
              f"release={args.compress_release:.0f} ms")
    if args.tone:
        _shelf_hz = args.tone_shelf_hz   if args.tone_shelf_hz   is not None else 220.0
        _shelf_db = args.tone_shelf_db   if args.tone_shelf_db   is not None else 3.5
        _hcut_hz  = args.tone_highcut_hz if args.tone_highcut_hz is not None else 7500.0
        print(f"  tone     : shelf={_shelf_hz:.0f} Hz {_shelf_db:+.1f} dB  "
              f"highcut={_hcut_hz:.0f} Hz")
    if args.no_phase_rand:
        print(f"  phase_rand: off")
    elif args.phase_rand is not None:
        print(f"  phase_rand: {args.phase_rand:.1f} ms (all layers)")
    elif args.warmth:
        print(f"  phase_rand: 10.0 ms (warmth default)")
    if args.midi:
        print(f"  midi     : {args.midi}")
    if args.seed is not None:
        print(f"  seed     : {args.seed}")
    print()

    rate, audio = load_wav(str(in_path))
    dur_in = audio.shape[0] / rate
    ch_str = "stereo" if audio.ndim == 2 else "mono"
    print(f"  Loaded   : {dur_in:.2f} s  {ch_str}  @ {rate} Hz")
    stretch_out = dur_in * (1 + args.stretch / 100)
    print(f"  Target   : {stretch_out:.1f} s" +
          (f"  + {reverb.decay_seconds():.1f}s reverb tail" if args.reverb else ""))
    print()

    stretcher = LaminarStretcher(
        stretch_percent = args.stretch,
        layers          = layers,
        auto_grain      = not args.no_auto_grain,
        fade_ms         = args.fade_ms,
        lfos            = lfos,
        chord           = chord,
        seed            = args.seed,
    )

    # stage 1: granular time-stretch
    result = stretcher.process(audio, rate)

    # stage 2: reverb
    if args.reverb:
        print("\nReverb")
        result = reverb.process(result, rate)

    # stage 3: compression
    if args.compress:
        compressor = LaminarCompressor(
            threshold_db = args.compress_threshold,
            ratio        = args.compress_ratio,
            makeup_db    = args.compress_makeup,
            release_ms   = args.compress_release,
        )
        print(f"\nCompress  threshold={args.compress_threshold:.0f} dBFS  "
              f"ratio={args.compress_ratio:.1f}:1  "
              f"makeup={args.compress_makeup:+.0f} dB  "
              f"release={args.compress_release:.0f} ms")
        result = compressor.process(result, rate)

    # stage 4: tone correction
    if args.tone:
        shelf_hz   = args.tone_shelf_hz   if args.tone_shelf_hz   is not None else 220.0
        shelf_db   = args.tone_shelf_db   if args.tone_shelf_db   is not None else 3.5
        highcut_hz = args.tone_highcut_hz if args.tone_highcut_hz is not None else 7500.0
        print(f"\nTone  shelf={shelf_hz:.0f} Hz {shelf_db:+.1f} dB  highcut={highcut_hz:.0f} Hz")
        tone = ToneCorrector(rate, shelf_hz=shelf_hz, shelf_db=shelf_db, highcut_hz=highcut_hz)
        result = tone.process(result)

    # stage 5: peak normalisation
    if not args.no_normalize:
        peak = np.max(np.abs(result))
        if peak > 1e-8:
            result /= peak
    else:
        result = np.clip(result, -1.0, 1.0)

    save_wav(args.output, rate, result)
    dur_out = result.shape[0] / rate
    print(f"\n  Saved    : {dur_out:.2f} s  ->  {args.output}")

    if args.midi:
        print(f"\nPitch analysis  ->  {args.midi}")
        print(f"  fmin={args.midi_fmin:.0f} Hz  fmax={args.midi_fmax:.0f} Hz  "
              f"confidence>={args.midi_confidence:.2f}  min_note={args.midi_min_note:.2f}s")
        analyzer = PitchAnalyzer(
            fmin         = args.midi_fmin,
            fmax         = args.midi_fmax,
            frame_ms     = args.midi_frame,
            hop_ms       = args.midi_hop,
            confidence   = args.midi_confidence,
            min_note_s   = args.midi_min_note,
        )
        notes = analyzer.analyze(result, rate)
        if notes and args.midi_merge:
            notes = _merge_midi_notes(notes, args.midi_merge_gap / 1000.0)
        if notes:
            analyzer.write_midi(notes, args.midi, bpm=args.midi_bpm)
            note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
            summary = ', '.join(
                f"{note_names[n % 12]}{n // 12 - 1}({t:.1f}s)"
                for t, _, n, _ in notes[:8]
            )
            print(f"  {len(notes)} note{'s' if len(notes) != 1 else ''}  "
                  f"first 8: {summary}{'...' if len(notes) > 8 else ''}")
            print(f"  Saved MIDI -> {args.midi}")
        else:
            print("  No pitched notes detected -- try lowering --midi-confidence")

    print("=" * 58)


if __name__ == "__main__":
    main()
