# toSinging: convert any speaking voice into a singing voice

任意の話し声を歌声に変換する言語非依存の音声変換システムです。

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Git LFS (for model file)

## Installation

```bash
# Clone repository with model file
git lfs install
git clone https://github.com/akinori-ito/toSinging.git
cd toSinging

# Install dependencies
uv sync
```

## Usage

```bash
uv run python tosinging.py -i input.wav -m input.musicxml [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-i` | Input WAV file (required) | - |
| `-m` | MusicXML file (required) | - |
| `-o` | Output WAV file | `output.wav` |
| `-modelfile` | VUV model file | `hubert_sad_20ms_model.pth` |
| `-modelname` | HuBERT model name | `facebook/hubert-base-ls960` |
| `-bpm` | Tempo (overrides score) | - |
| `-trans` | Transpose (semitones) | `0` |

### Example

```bash
uv run python tosinging.py -i speech.wav -m song.musicxml -o singing.wav -bpm 120
```

## Reference

A. Ito, Language Independent Speech-to-Singing-Voice Conversion. Jxiv preprint, https://doi.org/10.51094/jxiv.1902
