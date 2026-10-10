# Valorant Voice ASR Pipeline

Extract Korean transcripts from all 12,823 Korean VO .wav files using faster-whisper.

## Quick Start (GitHub Actions)

1. Fork this repo (must be **public** for unlimited Actions minutes)
2. Put your `.wav` files in `sounds_kr/raw/`
3. Go to **Actions** → **ASR Full Pipeline** → **Run workflow**
   - `model`: `small` (recommended), `base`, `medium`, `large-v3`
   - `chunk`: `0` (0-indexed)
   - `total_chunks`: `20` (or more for faster parallel)
3. Click **Run workflow**
4. Wait ~30 min (20 parallel jobs on GitHub Actions)
5. Download `asr-result-chunk-*` artifacts → merge → `sounds_kr/asr_full.json`

## Local / Codespaces

```bash
# Codespace opens with devcontainer (32-core, 64GB RAM)
python asr_full.py  # set CHUNK=0 TOTAL_CHUNKS=1 for single run
```

## Output

`sounds_kr/asr_full.json` — mapping `filename.wav` → `{text, dur}`

```json
{
  "1000042277.wav": {"text": "이름 괜찮아", "dur": 1.38},
  "1000133397.wav": {"text": "??..", "dur": 0.87}
}
```

## Merge chunks locally

```bash
python -c "
import json, glob
d = {}
for f in glob.glob('asr-result-chunk-*/sounds_kr/asr_full.json'):
    d.update(json.load(open(f)))
json.dump({k:v for f in glob.glob('asr-result-chunk-*') for k,v in json.load(open(f)).items()}, open('sounds_kr/asr_full.json','w'), ensure_ascii=False)
"
```

## Files

- `asr_full.py` — main processing script (chunk-aware)
- `requirements.txt` — Python deps
- `.devcontainer/` — Codespaces config (32-core, 64GB RAM)