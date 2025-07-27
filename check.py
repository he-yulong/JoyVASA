import soundfile as sf
import os

files = [
    'data/RD_Radio1_000.wav',
    'data/RD_Radio2_000.wav',
]
for f in files:
    try:
        data, sr = sf.read(f)
        print(f"✅ {f}: shape={data.shape}, sr={sr}")
    except Exception as e:
        print(f"❌ {f}: {e}")

import json
with open('data/train.json') as f:
    d = json.load(f)
    for entry in d:
        print(entry['audio_name'], os.path.exists(entry['audio_name']))

import pickle

with open('data/motions.pkl', 'rb') as f:
    motions = pickle.load(f)
print(list(motions.keys()))