import json
import os
import torchaudio
import numpy as np
import torch
from torch.utils import data
import pickle
import warnings

torchaudio.set_audio_backend('soundfile')
warnings.filterwarnings('ignore', message='PySoundFile failed. Trying audioread instead.')


class TalkingHeadDatasetHungry(data.Dataset):
    def __init__(self, root_dir, motion_filename="talking_face.pkl", motion_template_filename="motion_template.pkl",
                 split="train", coef_fps=25, n_motions=100, crop_strategy="random", normalize_type="mix"):
        self.templete_dir = os.path.join(root_dir, motion_template_filename)
        self.templete_dict = pickle.load(open(self.templete_dir, 'rb'))
        self.motion_dir = os.path.join(root_dir, motion_filename)
        self.eps = 1e-9
        self.normalize_type = normalize_type
        self.n_motions = n_motions
        self.coef_total_len = self.n_motions * 2

        if split == "train":
            self.root_dir = os.path.join(root_dir, "train.json")
        else:
            self.root_dir = os.path.join(root_dir, "test.json")

        with open(self.root_dir, 'r') as f:
            json_data = json.load(f)
        self.motion_data = pickle.load(open(self.motion_dir, "rb"))
        print("load all motion data done...")
        # Required number of frames for one sample
        min_frames = self.coef_total_len + 2
        # Keep only items with enough motion frames
        self.all_data = []
        for item in json_data:
            motion = self.motion_data[item["audio_name"]]
            if motion["n_frames"] >= min_frames:
                self.all_data.append(item)
            else:
                print(f"Skipping {item['audio_name']} - too short ({motion['n_frames']} frames)")

        self.coef_fps = coef_fps
        self.audio_unit = 16000. / self.coef_fps  # num of samples per frame
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.audio_total_len = round(self.audio_unit * self.coef_total_len)
        self.crop_strategy = crop_strategy

    def __len__(self, ):
        return len(self.all_data)

    def _select_crop_window(self, seq_len: int) -> tuple[int, int] | None:
        """
        Selects a contiguous frame window [start_frame, end_frame)
        from a sequence based on the configured crop strategy.

        Returns:
            (start_frame, end_frame) if valid, else None if sequence too short.
        """
        # Check if sequence is long enough
        if seq_len < self.coef_total_len:
            return None  # invalid

        if self.crop_strategy == 'random':
            max_start = seq_len - self.coef_total_len
            start_frame = np.random.randint(0, max_start + 1)

        elif self.crop_strategy == 'begin':
            start_frame = 0

        elif self.crop_strategy == 'end':
            start_frame = seq_len - self.coef_total_len

        else:
            raise ValueError(f'Unknown crop strategy: {self.crop_strategy}')

        end_frame = start_frame + self.coef_total_len
        return start_frame, end_frame

    def _normalize_motion_window(self, motion_data, start_frame, end_frame):
        """
        Extracts and normalizes exp and pose features from motion_data
        for frames in [start_frame, end_frame).

        Returns:
            coef_dict: dict with "exp" and "pose" as torch tensors.
        """
        coef_keys = ["exp", "pose"]
        coef_dict = {k: [] for k in coef_keys}

        for frame_idx in range(start_frame, end_frame):
            frame = motion_data['motion'][frame_idx]

            # Normalize expression (z-score)
            if self.normalize_type != "mix":
                raise RuntimeError("Unsupported normalize_type for exp/pose")

            normalized_exp = (
                                     frame["exp"].flatten() - self.templete_dict["mean_exp"]
                             ) / (self.templete_dict["std_exp"] + self.eps)
            coef_dict["exp"].append([normalized_exp])

            # Normalize pose (min–max per component)
            pose_data = np.concatenate((
                (frame["scale"].flatten() - self.templete_dict["min_scale"]) /
                (self.templete_dict["max_scale"] - self.templete_dict["min_scale"] + self.eps),
                (frame["t"].flatten() - self.templete_dict["min_t"]) /
                (self.templete_dict["max_t"] - self.templete_dict["min_t"] + self.eps),
                (frame["pitch"].flatten() - self.templete_dict["min_pitch"]) /
                (self.templete_dict["max_pitch"] - self.templete_dict["min_pitch"] + self.eps),
                (frame["yaw"].flatten() - self.templete_dict["min_yaw"]) /
                (self.templete_dict["max_yaw"] - self.templete_dict["min_yaw"] + self.eps),
                (frame["roll"].flatten() - self.templete_dict["min_roll"]) /
                (self.templete_dict["max_roll"] - self.templete_dict["min_roll"] + self.eps),
            ))
            coef_dict["pose"].append([pose_data])

        # Convert to torch tensors
        coef_dict = {k: torch.tensor(np.concatenate(v, axis=0)) for k, v in coef_dict.items()}
        return coef_dict

    def _load_and_crop_audio(self, audio_path, start_frame, end_frame):
        """
        Loads audio from file, crops it to match the given frame range,
        and ensures the length matches the expected duration.

        Returns:
            audio (1D torch.Tensor) if valid, else None.
        """
        # Load
        audio_clip, sr = torchaudio.load(audio_path)
        audio_clip = audio_clip.squeeze()

        # Sample rate check
        if sr != 16000:
            raise ValueError(f"Invalid sampling rate: {sr}, expected 16000")

        # Convert frames to samples
        start_sample = round(start_frame * self.audio_unit)
        end_sample = round(end_frame * self.audio_unit)

        # Crop
        audio = audio_clip[start_sample:end_sample]

        # Validate length
        expected_len = int(self.coef_total_len * self.audio_unit)
        if audio.shape[0] != expected_len:
            print(f"audio length invalid! audio: {audio.shape[0]}, coef: {expected_len}")
            return None  # invalid

        return audio

    def _split_into_pairs(self, audio: torch.Tensor, coef_dict: dict):
        """
        Splits the full audio & coefficient chunk into two equal halves:
        (A) first n_motions frames/samples
        (B) last n_motions frames/samples

        Args:
            audio: 1D torch.Tensor of shape [coef_total_len * audio_unit]
            coef_dict: dict with 'exp' and 'pose' tensors, each shape [coef_total_len, feat_dim]

        Returns:
            audio_pair: [audio_A, audio_B]
            coef_pair:  [{exp_A, pose_A}, {exp_B, pose_B}]
        """
        keys = ['exp', 'pose']

        # Audio split
        audio_pair = [
            audio[:self.n_audio_samples].clone(),
            audio[-self.n_audio_samples:].clone()
        ]

        # Coefficient split
        coef_pair = [
            {k: coef_dict[k][:self.n_motions].clone() for k in keys},
            {k: coef_dict[k][-self.n_motions:].clone() for k in keys}
        ]

        return audio_pair, coef_pair

    def __getitem__(self, index):
        """
        For each index i, it returns two aligned halves of one sample:
            - audio_pair: a list of two 1‑D tensors [audio_A, audio_B]
                - Each is a contiguous chunk of waveform (shape ≈ n_audio_samples, sampled at 16 kHz)
                - audio_A = first half; audio_B = second half
            - coef_pair: a list of two dicts
                - coef_pair[0] = {'exp': exp_A, 'pose': pose_A}
                - coef_pair[1] = {'exp': exp_B, 'pose': pose_B}
                - Each exp_* has shape [n_motions, D_exp]
                - Each pose_* has shape [n_motions, D_pose] where pose = concat(scale, t, pitch, yaw, roll)

        A and B are time‑adjacent. Audio and motion are aligned frame‑by‑frame using the FPS you give (coef_fps, default 25 FPS).
        """
        has_valid_audio = False
        while not has_valid_audio:
            # read motion
            metadata = self.all_data[
                index]  # {'video_name': '.../RD_Radio1_000.mp4', 'audio_name': '.../RD_Radio1_000.wav', 'motion_name': '.../RD_Radio1_000.pkl'}
            motion_data = self.motion_data[metadata[
                "audio_name"]]  # {'motion': list, 'c_eyes_lst': list, 'c_lip_lst': list, 'n_frames': int, 'output_fps': 29}
            seq_len = motion_data["n_frames"]
            start_frame, end_frame = self._select_crop_window(seq_len)

            coef_dict = self._normalize_motion_window(motion_data, start_frame, end_frame)
            assert coef_dict['exp'].shape[0] == self.coef_total_len, f"Invalid coef length: {coef_dict['exp'].shape[0]}"

            audio = self._load_and_crop_audio(metadata["audio_name"], start_frame, end_frame)
            if audio is None:
                has_valid_audio = False
                continue

            audio_pair, coef_pair = self._split_into_pairs(audio, coef_dict)
            has_valid_audio = True

            return audio_pair, coef_pair


if __name__ == "__main__":
    data_root = "../../data"
    motion_filename = "motions.pkl"
    motion_template_filename = "motion_template.pkl"
    normalize_type = "mix"

    train_dataset = TalkingHeadDatasetHungry(data_root,
                                             motion_filename=motion_filename,
                                             motion_template_filename=motion_template_filename,
                                             split="train", coef_fps=25, n_motions=100, crop_strategy="random",
                                             normalize_type=normalize_type)
    train_loader = data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=8, pin_memory=True)
    for audio_pair, coef_pair in train_loader:
        print(f"audio: {audio_pair[0].shape}, {audio_pair[1].shape}, exp: {coef_pair[0]['exp'].shape}, \
              {coef_pair[1]['exp'].shape}, pose: {coef_pair[0]['pose'].shape}, {coef_pair[1]['pose'].shape}")
