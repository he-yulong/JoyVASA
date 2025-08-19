import os
import tyro
import multiprocessing
import sys
import subprocess

sys.path.append(os.path.dirname(os.path.abspath("../")))

from joyvasa.config.argument_config import ArgumentConfig
from joyvasa.motion_extractor import make_motion_templete


def extract_audio_if_needed(video_path, wav_path):
    if not os.path.exists(wav_path):
        print(f"🔊 Extracting audio for: {os.path.basename(video_path)}")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000", wav_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
        except subprocess.CalledProcessError:
            print(f"❌ Failed to extract audio from {video_path}")


def process_videos(args, video_list, suffix):
    params = [(args, driving_video, suffix) for driving_video in video_list]
    with multiprocessing.Pool(processes=2) as pool:
        # For each .mp4 in root_dir, it calls make_motion_templete
        # This will produce a .pkl file alongside each video
        # The .pkl contains the motion template: mean/std/min/max
        # for each motion coefficient (exp, scale, translation, pitch, yaw, roll) computed from that video.
        pool.starmap(make_motion_templete, params)


def main():
    args = tyro.cli(ArgumentConfig)
    args.flag_do_crop = False  # don’t crop face images when extracting motion.
    args.scale = 2.3  # scaling factor for bounding boxes / image resizing during motion extraction.

    root_dir = "D:/dev/jupyter_book/JoyVASA/data"
    video_names = sorted([os.path.join(root_dir, filename)
                          for filename in os.listdir(root_dir)
                          if filename.lower().endswith("mp4")])

    # Step 1: Ensure each .wav file exists
    for video_path in video_names:
        wav_path = os.path.splitext(video_path)[0] + ".wav"
        extract_audio_if_needed(video_path, wav_path)

    # Step 2: Run motion template generation
    process_videos(args, video_names, suffix=".pkl")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
