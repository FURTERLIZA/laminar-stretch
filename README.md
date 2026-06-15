# laminar stretch

granular time-stretching for ambient and drone production, written in Python. processes `.wav` files at extreme stretch ratios (200%–800%+).

takes a source audio file and slows it down by an order of magnitude without pitch-shifting. unlike single-pass granular stretching (paulstretch-style), it runs several independent stretch passes in parallel — each with a different grain size, pitch offset, and rate — and mixes the results. the layers evolve at slightly different speeds and beat against each other, producing movement rather than a static smear. optional post-processing covers reverb, compression, tone correction, LFO modulation, chord harmonisation, and pitch analysis with MIDI export.

## features

- multi-layer granular overlap-add engine (up to 4 independent layers)
- per-layer grain size, stretch rate, pitch offset, jitter, and phase randomisation
- LFO modulation of grain size, jitter, or gain (sine, triangle, random walk)
- chord mode: duplicates the layer stack at harmonic intervals and mixes
- procedural convolution reverb with stereo width control
- soft-knee RMS compression with IIR release
- tone correction: low-shelf boost and Butterworth high cut
- `--warmth` preset for ambient material
- YIN pitch detection with MIDI export and note merging
- single-file Python script, no compiled dependencies

## approach

two ideas drive the design:

**pizza dough:** stretch in several separate pulls rather than one long yank. a single extreme stretch tears the material; repeated gentler passes produce something coherent and evolving.

**pastry lamination:** process multiple independent layers and fold them together. each layer has its own grain size, stretch modifier, and subtle pitch offset. because they evolve at slightly different rates they beat against each other, producing natural movement rather than static repetition.

yes. i was hungry when i thought of this.

the core technique is overlap-add granular synthesis. the source file is scanned slowly (`hop_in << hop_out`) and windowed segments are placed into the output. each output grain is a Hann-weighted average of several nearby source grains, placed at a scattered position. post-processing stages (reverb, compression, tone correction) are applied to the mixed output in that order.

see [algorithm.md](algorithm.md) for a step-by-step breakdown.

this is a reference implementation in Python.

## requirements

```
numpy>=1.24
scipy>=1.10
```

```
pip install -r requirements.txt
```

---

## basic usage

```
python laminar_stretch.py input.wav output.wav --stretch 800
python laminar_stretch.py input.wav output.wav --stretch 400 --reverb --warmth
python laminar_stretch.py input.wav output.wav --stretch 200 --midi out.mid
```

`--stretch 100` doubles the length. `--stretch 800` produces 9x the original duration.

---

## command-line options

### stretch

| flag | default | description |
|---|---|---|
| `-s`, `--stretch PCT` | `200` | stretch amount as a percentage. 100 = 2x length, 800 = 9x length. |
| `-l`, `--num-layers N` | `4` | number of layers (1–4). |
| `--grain-ms MS` | | override base grain size in ms for all layers. |
| `--no-auto-grain` | off | disable automatic grain scaling at extreme stretch ratios. |
| `--phase-rand MS` | | set phase_rand_ms on all layers. |
| `--no-phase-rand` | off | disable phase randomisation on all layers. |
| `--warmth` | off | preset: grain-ms 280, tone correction on, phase-rand 10 ms. explicit flags override each part. |
| `--fade-ms MS` | `30` | cosine fade-in/out length in ms. |
| `--seed N` | | random seed. |
| `--no-normalize` | off | skip peak normalisation. |

### reverb

| flag | default | description |
|---|---|---|
| `--reverb` | off | enable convolution reverb. always outputs stereo. |
| `--reverb-room 0–1` | `0.65` | controls decay time (0 = 0.5 s, 1 = 8 s). |
| `--reverb-damp 0–1` | `0.35` | high-frequency absorption in the tail. |
| `--reverb-wet 0–1` | `0.25` | wet signal level. |
| `--reverb-width 0–1` | `0.80` | stereo spread of the reverb tail. |
| `--reverb-predelay MS` | `12` | silence before the first reflection. |

### compression

| flag | default | description |
|---|---|---|
| `--compress` | off | enable dynamic range compression. |
| `--compress-threshold dBFS` | `-20` | level above which gain reduction begins. |
| `--compress-ratio N` | `3.0` | compression ratio above threshold. |
| `--compress-makeup dB` | `3.0` | makeup gain added after compression. |
| `--compress-release MS` | `200` | gain recovery time after a loud passage. |

### tone correction

applied after compression. low shelf followed by a Butterworth high cut.

| flag | default | description |
|---|---|---|
| `--tone` | off | enable tone correction. |
| `--tone-shelf-hz HZ` | `220` | low-shelf corner frequency. |
| `--tone-shelf-db dB` | `+3.5` | low-shelf boost in dB. |
| `--tone-highcut-hz HZ` | `7500` | high-cut -3 dB point. |

### chord mode

copies the layer stack for each interval and mixes them together.

| flag | default | description |
|---|---|---|
| `--chord PRESET` | | `fifth` [0,7], `octave` [0,12], `minor` [0,3,7], `major` [0,4,7], `cluster` [0,1,5,8]. |
| `--chord-gain 0–1` | `0.4` | mix level for non-root intervals. |

### LFO modulation

modulates a layer parameter over time. repeat `--lfo` for multiple LFOs.

```
--lfo TARGET:WAVEFORM:RATE_HZ:DEPTH
```

- **TARGET:** `grain_ms`, `jitter_ms`, or `gain`
- **WAVEFORM:** `sine`, `triangle`, or `random_walk`
- **RATE_HZ:** modulation frequency in Hz
- **DEPTH:** proportion of the base value to displace (0.3 = 30%)

```
--lfo grain_ms:sine:0.08:0.3 --lfo gain:random_walk:0.05:0.2
```

### pitch analysis and MIDI export

analyses the output with the YIN algorithm and writes a type-0 MIDI file.

MIDI timing is tempo-relative. to align the MIDI with the output WAV in a DAW:

1. note what BPM your project is set to
2. pass that value as `--midi-bpm` when running the stretch
3. import both files at bar 1 beat 1

the default is 120 BPM, which is the default in most DAWs, so if you haven't changed your project tempo you don't need the flag.

| flag | default | description |
|---|---|---|
| `--midi PATH` | | write MIDI to this path. |
| `--midi-bpm BPM` | `120` | DAW project tempo in BPM. MIDI ticks are scaled to match. |
| `--midi-merge` | off | merge repeated same-pitch notes into single long notes. useful for drone material where the analyser produces short runs of the same pitch. |
| `--midi-merge-gap MS` | `500` | max gap between notes at the same pitch to be merged. only used with `--midi-merge`. |
| `--midi-fmin HZ` | `50` | lowest detectable pitch. |
| `--midi-fmax HZ` | `1000` | highest detectable pitch. |
| `--midi-confidence 0–1` | `0.60` | confidence threshold; lower accepts more frames. |
| `--midi-min-note S` | `0.15` | minimum note duration in seconds. |
| `--midi-frame MS` | `100` | analysis frame length. |
| `--midi-hop MS` | `50` | hop between analysis frames. |

---

## example

```
python laminar_stretch.py voice.wav out.wav \
    --stretch 800 \
    --warmth \
    --reverb --reverb-room 0.8 \
    --compress \
    --lfo grain_ms:random_walk:0.07:0.25 \
    --midi out.mid --midi-bpm 120
```

---

## github topics

suggested topics to add via repository settings:

`granular-synthesis` `time-stretching` `audio-processing` `ambient-music` `drone` `python` `dsp` `paulstretch` `sound-design` `midi` `wav` `overlap-add`
