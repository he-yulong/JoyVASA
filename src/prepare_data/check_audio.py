import soundfile as sf

try:
    data, sr = sf.read("../../data/RD_Radio1_000.wav")
    print(f"✅ Read successful: shape={data.shape}, sr={sr}")
except Exception as e:
    print(f"❌ Failed to read .wav: {e}")
