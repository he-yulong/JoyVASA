"""
Input: Raw videos (.mp4, etc.) in root_dir
Output: .wav files (mono, 16 kHz, uncompressed) alongside the videos
Why: Later steps (01_extract_motions.py, 02_gen_labels.py, etc.) will need audio files for:
- Audio-to-motion model training
- Feature extraction (e.g., via wav2vec2)
- Syncing motion and audio data
"""
import os
import subprocess
import multiprocessing


def is_video(file_path):
    if file_path.lower().endswith((".mp4", ".mov", ".avi", ".webm")) or os.path.isdir(file_path):
        return True
    return False

def prefix(filename):
    """a.jpg -> a"""
    pos = filename.rfind(".")
    if pos == -1:
        return filename
    return filename[:pos]

def basename(filename):
    """a/b/c.jpg -> c"""
    return prefix(os.path.basename(filename))

def remove_suffix(filepath):
    """a/b/c.jpg -> a/b/c"""
    return os.path.join(os.path.dirname(filepath), basename(filepath))

def extract_audio(filename, suffix=".wav"):
    """
    extract_audio() calls ffmpeg to extract the audio stream from the video:
    ffmpeg -i input.mp4 -vn -ar 16000 -ac 1 -c:a pcm_s16le output.wav
    -vn: ignore video
    -ar 16000: resample to 16 kHz (matches dataset code expectations)
    -ac 1: mono
    pcm_s16le: 16-bit signed little-endian PCM (uncompressed)
    """
    audio_output_name = remove_suffix(filename) + suffix
    if os.path.exists(audio_output_name):
        print("audio already generated~")
        return
    if os.path.exists(filename) and is_video(filename):
        subprocess.run(['ffmpeg', '-loglevel', 'quiet', '-i', filename, '-vn', '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', audio_output_name, '-y'])
    else:
        raise Exception(f"{filename} is not a supported type!")

def process_videos(video_list, suffix=".wav"):
    with multiprocessing.Pool(processes=12) as pool:
        pool.starmap(extract_audio, [(driving_video, suffix) for driving_video in video_list])


if __name__ == "__main__":
    root_dir = "D:/dev/jupyter_book/JoyVASA/data"
    video_names = sorted([os.path.join(root_dir, filename) for filename in os.listdir(root_dir) if filename.endswith("mp4")])
    process_videos(video_names, suffix=".wav")