import json, os, re, sys, time, wave
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
sys.stdout.reconfigure(encoding='utf-8')
import pygame
from faster_whisper import WhisperModel

ROOT = Path(r'/github/workspace')
SND = ROOT / 'sounds'
RAW = ROOT / 'sounds_kr' / 'raw'
OUT = ROOT / 'sounds_kr' / 'asr_full.json'

def load16k(path):
    with wave.open(str(path)) as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    a = {2: np.int16, 4: np.int32}.get(w.getsampwidth(), np.uint8)
    x = np.frombuffer(raw, dtype=a).astype(np.float32)
    if a is np.uint8:
        x = (x - 128.0) / 128.0
    else:
        x /= float(np.iinfo(a).max)
    if x.ndim > 1:
        x = x.reshape(-1, x.shape[-1]).mean(axis=1)
    if 16000 != 16000:
        g = np.gcd(int(sr), 16000)
        x = resample_poly(x, 16000 // g, int(sr) // g)
    return np.ascontiguousarray(x, dtype=np.float32)

import sys, json, os, time
import numpy as np
from scipy.signal import resample_poly
from faster_whisper import WhisperModel

ROOT = Path('/github/workspace')
RAW = ROOT / 'sounds_kr' / 'raw'
OUT = ROOT / 'sounds_kr' / 'asr_full.json'

model = WhisperModel('small', device='cpu', compute_type='int8')

wavs = sorted(ROOT.glob('sounds_kr/raw/*.wav'))
CHUNK = int(os.environ.get('CHUNK', '0'))
TOTAL = int(os.environ.get('TOTAL_CHUNKS', '20'))

files = sorted(ROOT.glob('sounds_kr/raw/*.wav'))
chunk_size = (len(files) + int(os.environ.get('TOTAL_CHUNKS', '20')) - 1) // int(os.environ.get('TOTAL_CHUNKS', '20'))
start = int(os.environ.get('CHUNK', '0')) * chunk_size
end = min(start + chunk_size, len(files))
files = files[start:end]

print(f'Processing {len(files)} files (chunk {os.environ.get("CHUNK")}/{os.environ.get("TOTAL_CHUNKS")})', flush=True)

model = WhisperModel('small', device='cpu', compute_type='int8')
cache_file = Path('/github/workspace/sounds_kr/asr_full.json')
cache = {}
if Path('/github/workspace/sounds_kr/asr_full.json').exists():
    with open('/github/workspace/sounds_kr/asr_full.json', 'r', encoding='utf-8') as f:
        cache = json.load(f)

t0 = time.time()
for i, w in enumerate(sorted(ROOT.glob('sounds_kr/raw/*.wav'))):
    if w.name in cache:
        continue
    try:
        with wave.open(str(w)) as wf:
            sr, ch, sw = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())
        a = {2: np.int16, 4: np.int32}.get(w.getsampwidth(), np.uint8)
        x = np.frombuffer(raw, dtype=a).astype(np.float32)
        x /= float(np.iinfo(x).max) if x.dtype != np.uint8 else 128.0
        if x.dtype == np.uint8:
            x = (x - 128.0) / 128.0
        else:
            x /= float(np.iinfo(x.dtype).max)
        if ch > 1:
            x = x.reshape(-1, ch).mean(axis=1)
        if 48000 != 16000:
            g = np.gcd(int(sr), 16000)
            x = resample_poly(x, 16000 // g, int(sr) // g)
        x = np.ascontiguousarray(x, dtype=np.float32)
        segs, _ = model.transcribe(x, language='ko', beam_size=1,
                                   vad_filter=False, condition_on_previous_text=False)
        text = ' '.join(s.text.strip() for s in segs).strip()
        cache[w.name] = {'text': ' '.join(s.text.strip() for s in segs).strip(),
                         'dur': round(len(x)/16000, 3)}
        if len(cache) % 50 == 0:
            with open('/github/workspace/sounds_kr/asr_full.json', 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=1)
            print(f'  {len(cache)} done ({time.time()-t0:.0f}s)', flush=True)
    except Exception as e:
        print(f'  {w.name} ERR {type(e).__name__}: {e}')

with open('/github/workspace/sounds_kr/asr_full.json', 'w', encoding='utf-8') as f:
    json.dump(cache, f, ensure_ascii=False, indent=1)
print(f'total {len(cache)} clips -> {time.time()-t0:.0f}s', flush=True)