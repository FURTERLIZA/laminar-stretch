#!/usr/bin/env python3
"""
Quick smoke test: generate a short sine-wave .wav, stretch it at several
factors, and verify basic output properties (duration, peak, stereo shape).
"""

import numpy as np
import scipy.io.wavfile as wavfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fl_stretch import LaminarStretcher, LaminarReverb, LaminarCompressor, DEFAULT_LAYERS, load_wav, save_wav


RATE = 44100


def make_test_wav(path: str, freq: float = 220.0, duration: float = 2.0) -> None:
    t = np.linspace(0, duration, int(RATE * duration), endpoint=False)
    wave = 0.6 * np.sin(2 * np.pi * freq * t) + 0.4 * np.sin(2 * np.pi * freq * 1.5 * t)
    wavfile.write(path, RATE, (wave * 32767).astype(np.int16))


def run_stretch(stretch_pct: float, num_layers: int, auto_grain: bool, label: str) -> None:
    src = "/tmp/fl_smoke_in.wav"
    dst = f"/tmp/fl_smoke_out_{label}.wav"
    make_test_wav(src)

    rate, audio = load_wav(src)
    dur_in = audio.shape[0] / rate

    stretcher = LaminarStretcher(
        stretch_percent=stretch_pct,
        layers=DEFAULT_LAYERS[:num_layers],
        auto_grain=auto_grain,
        fade_ms=30.0,
        seed=0,
    )
    result = stretcher.process(audio, rate)
    save_wav(dst, rate, result)

    dur_out  = result.shape[0] / rate
    expected = dur_in * (1 + stretch_pct / 100)
    assert abs(dur_out - expected) < 0.1, f"Duration mismatch: {dur_out:.3f} vs {expected:.3f}"
    assert np.max(np.abs(result)) <= 1.0 + 1e-6
    print(f"  PASS  [{label}]  {dur_in:.1f}s → {dur_out:.1f}s  (×{dur_out/dur_in:.2f})")


def run_compress(label: str) -> None:
    rate = 44100
    # Signal with intentionally wide dynamic range: loud then quiet
    loud  = np.sin(2 * np.pi * 220 * np.linspace(0, 2, rate * 2)) * 0.9
    quiet = np.sin(2 * np.pi * 220 * np.linspace(0, 2, rate * 2)) * 0.05
    audio = np.concatenate([loud, quiet]).astype(np.float64)

    comp = LaminarCompressor(threshold_db=-20, ratio=3.0, makeup_db=3.0, release_ms=200)
    result = comp.process(audio, rate)

    loud_rms_before  = np.sqrt(np.mean(audio[:rate * 2] ** 2))
    quiet_rms_before = np.sqrt(np.mean(audio[rate * 2:] ** 2))
    loud_rms_after   = np.sqrt(np.mean(result[:rate * 2] ** 2))
    # Use the last half of the quiet section, where release is complete
    quiet_rms_after  = np.sqrt(np.mean(result[rate * 3:] ** 2))

    ratio_before = loud_rms_before / quiet_rms_before
    ratio_after  = loud_rms_after  / quiet_rms_after

    assert loud_rms_after < loud_rms_before, "Loud section should be attenuated"
    assert ratio_after < ratio_before, "Loud/quiet ratio should narrow"
    assert np.max(np.abs(result)) <= np.max(np.abs(audio)) * 2.0, "Output shouldn't explode"
    print(f"  PASS  [{label}]  loud/quiet ratio {ratio_before:.1f}× → {ratio_after:.1f}×")


def run_reverb(label: str) -> None:
    src = "/tmp/fl_smoke_in.wav"
    make_test_wav(src)
    rate, audio = load_wav(src)

    # Stretch first
    stretcher = LaminarStretcher(stretch_percent=100, layers=DEFAULT_LAYERS[:1], seed=0)
    stretched = stretcher.process(audio, rate)

    # Reverb — mono in, stereo out
    reverb = LaminarReverb(room_size=0.6, damping=0.4, wet=0.3, dry=1.0, width=0.8, seed=42)
    result = reverb.process(stretched, rate)

    assert result.ndim == 2 and result.shape[1] == 2, "Reverb output should be stereo"
    assert result.shape[0] > stretched.shape[0], "Reverb should extend output with tail"
    assert np.max(np.abs(result)) <= 1.0 + 1e-6
    dur_out = result.shape[0] / rate
    print(f"  PASS  [{label}]  stereo={result.shape[1]}ch  dur={dur_out:.1f}s")


if __name__ == "__main__":
    print("Laminar Stretch — smoke tests")
    print("─" * 50)
    run_stretch(100,  1, False, "2x_1layer_no_autograin")
    run_stretch(200,  4, True,  "3x_4layer_autograin")
    run_stretch(400,  4, True,  "5x_4layer_autograin")
    run_stretch(800,  4, True,  "9x_4layer_autograin")
    run_compress("compressor_narrows_dynamic_range")
    run_reverb("reverb_mono_in_stereo_out")
    print("─" * 50)
    print("All tests passed.")
