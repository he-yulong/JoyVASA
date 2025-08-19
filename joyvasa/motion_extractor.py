# coding: utf-8
import torch
torch.backends.cudnn.benchmark = True  # disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warning
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np
import os
import os.path as osp
import subprocess
from rich.progress import track

from .config.argument_config import ArgumentConfig
from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .utils.cropper import Cropper
from .utils.camera import get_rotation_matrix
from .utils.video import get_fps
from .utils.io import load_video, dump
from .utils.helper import is_video, is_template, remove_suffix, is_square_video
from .utils.rprint import rlog as log
from .live_portrait_wmg_wrapper import LivePortraitWrapper

import pathlib

pathlib.PosixPath = pathlib.WindowsPath  # temporary：change PosixPath to Path on Windows


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


def fast_check_args(args: ArgumentConfig):
    if not osp.exists(args.reference):
        raise FileNotFoundError(f"reference info not found: {args.reference}")
    if not osp.exists(args.driving):
        raise FileNotFoundError(f"driving info not found: {args.driving}")


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


"""
runs the LivePortrait model to extract keypoints & features
"""


class LivePortraitMotionExtractor(object):

    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.live_portrait_wrapper: LivePortraitWrapper = LivePortraitWrapper(inference_cfg=inference_cfg)
        self.cropper: Cropper = Cropper(crop_cfg=crop_cfg)

    def make_motion_template(self, I_lst, c_eyes_lst, c_lip_lst, **kwargs):
        n_frames = I_lst.shape[0]
        template_dct = {
            'n_frames': n_frames,
            'output_fps': kwargs.get('output_fps', 25),
            'motion': [],
            'c_eyes_lst': [],
            'c_lip_lst': [],
        }

        for i in track(range(n_frames), description='Making motion templates...', total=n_frames):
            # collect s, R, δ and t for inference
            I_i = I_lst[i]
            x_i_info = self.live_portrait_wrapper.get_kp_info(I_i)
            x_s = self.live_portrait_wrapper.transform_keypoint(x_i_info)
            R_i = get_rotation_matrix(x_i_info['pitch'], x_i_info['yaw'], x_i_info['roll'])

            item_dct = {
                'scale': x_i_info['scale'].cpu().numpy().astype(np.float32),
                'R': R_i.cpu().numpy().astype(np.float32),
                'exp': x_i_info['exp'].cpu().numpy().astype(np.float32),
                't': x_i_info['t'].cpu().numpy().astype(np.float32),
                'kp': x_i_info['kp'].cpu().numpy().astype(np.float32),
                'x_s': x_s.cpu().numpy().astype(np.float32),
                'pitch': x_i_info['pitch'].cpu().numpy().astype(np.float32),
                'yaw': x_i_info['yaw'].cpu().numpy().astype(np.float32),
                'roll': x_i_info['roll'].cpu().numpy().astype(np.float32)
            }

            template_dct['motion'].append(item_dct)

            c_eyes = c_eyes_lst[i].astype(np.float32)
            template_dct['c_eyes_lst'].append(c_eyes)

            c_lip = c_lip_lst[i].astype(np.float32)
            template_dct['c_lip_lst'].append(c_lip)

        return template_dct

    def _load_driving_video(self, driving_path):
        """
        Load a driving video if it exists and is supported.

        Returns:
            driving_rgb_lst (list of frames)
            output_fps (int)
            flag_is_driving_video (bool)
        Raises:
            FileNotFoundError: if the path does not exist
            Exception: if the file type is not supported
        """
        if not osp.exists(driving_path):
            raise FileNotFoundError(f"{driving_path} does not exist!")

        if is_video(driving_path):
            flag_is_driving_video = True
            # load from video file, AND make motion template
            output_fps = int(get_fps(driving_path))
            log(f"Load driving video from: {driving_path}, FPS is {output_fps}")
            driving_rgb_lst = load_video(driving_path)
            return driving_rgb_lst, output_fps, flag_is_driving_video
        else:
            raise Exception(f"{driving_path} is not a supported type!")

    def _should_crop(self, inf_cfg: InferenceConfig, driving_path: str) -> bool:
        """Decide whether to crop based on config or video shape."""
        return bool(inf_cfg.flag_crop_driving_video or (not is_square_video(driving_path)))

    def _crop_video(self, frames: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Crop driving frames and return (cropped_frames, cropped_landmarks).
        """
        ret = self.cropper.crop_driving_video(frames)
        log(f'Driving video is cropped, {len(ret["frame_crop_lst"])} frames are processed.')
        return ret['frame_crop_lst'], ret['lmk_crop_lst']

    def _compute_landmarks_no_crop(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """
        Compute landmarks from already-cropped or uncropped frames when skipping cropping.
        """
        return self.cropper.calc_lmks_from_cropped_video(frames)

    def _resize_to_256(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Resize all frames to 256x256."""
        return [cv2.resize(f, (256, 256)) for f in frames]

    def _adjust_frame_count(self, n_frames: int, processed_frames: int, is_video: bool) -> int:
        """
        Keep frame count consistent if cropping changed the number of frames.
        """
        if is_video and processed_frames != n_frames:
            return min(n_frames, processed_frames)
        return n_frames

    def _prepare_driving_frames(
            self,
            driving_rgb_lst: list[np.ndarray],
            inf_cfg: InferenceConfig,
            driving_path: str,
            flag_is_driving_video: bool,
    ) -> tuple[list[np.ndarray], list[np.ndarray], int]:
        """
        Decide crop, compute landmarks, resize to 256x256, and adjust n_frames.

        Returns:
            driving_rgb_crop_256x256_lst: list[np.ndarray] of 256x256 frames
            driving_lmk_crop_lst:          list[np.ndarray] landmarks per frame
            n_frames:                      int, possibly adjusted after cropping
        """
        n_frames = len(driving_rgb_lst)

        need_crop = bool(inf_cfg.flag_crop_driving_video or (not is_square_video(driving_path)))
        if need_crop:
            print("croping:", inf_cfg.flag_crop_driving_video)
            ret = self.cropper.crop_driving_video(driving_rgb_lst)
            frame_crop_lst, lmk_crop_lst = ret["frame_crop_lst"], ret["lmk_crop_lst"]
            log(f'Driving video is cropped, {len(frame_crop_lst)} frames are processed.')

            # Adjust frame count if cropping changed it
            if flag_is_driving_video and len(frame_crop_lst) != n_frames:
                n_frames = min(n_frames, len(frame_crop_lst))

            driving_rgb_crop_256x256_lst = [cv2.resize(f, (256, 256)) for f in frame_crop_lst]
            driving_lmk_crop_lst = lmk_crop_lst
        else:
            print("without crop ...")
            driving_lmk_crop_lst = self.cropper.calc_lmks_from_cropped_video(driving_rgb_lst)
            driving_rgb_crop_256x256_lst = [cv2.resize(f, (256, 256)) for f in driving_rgb_lst]  # force 256x256
        # End result:
        # You always end up with:
        # driving_rgb_crop_256x256_lst: resized frames (all 256×256).
        # riving_lmk_crop_lst: landmarks (whether from cropped or raw frames).
        return driving_rgb_crop_256x256_lst, driving_lmk_crop_lst, n_frames

    def execute(self, args, suffix=".pkl"):
        # Just a convenience handle to the wrapper’s inference config (e.g., crop flags, sizes).
        inf_cfg = self.live_portrait_wrapper.inference_cfg

        ######## process driving info ########
        wfp_template = remove_suffix(args.driving) + suffix
        if os.path.exists(wfp_template):
            log("motion generated ...")
            return

        driving_rgb_lst, output_fps, flag_is_driving_video = self._load_driving_video(args.driving)
        ######## make motion template ########
        log("Start making driving motion template...")
        driving_rgb_crop_256x256_lst, driving_lmk_crop_lst, _ = self._prepare_driving_frames(
            driving_rgb_lst=driving_rgb_lst,
            inf_cfg=inf_cfg,
            driving_path=args.driving,
            flag_is_driving_video=flag_is_driving_video,
        )
        c_d_eyes_lst, c_d_lip_lst = self.live_portrait_wrapper.calc_ratio(driving_lmk_crop_lst)
        # save the motion template
        I_d_lst = self.live_portrait_wrapper.prepare_videos(driving_rgb_crop_256x256_lst)
        driving_template_dct = self.make_motion_template(I_d_lst, c_d_eyes_lst, c_d_lip_lst, output_fps=output_fps)

        wfp_template = remove_suffix(args.driving) + suffix
        dump(wfp_template, driving_template_dct)
        log(f"Dump motion template to {wfp_template}")


# the entry point
def make_motion_templete(args, driving_video, suffix=".pkl"):
    # configs
    args.driving = driving_video
    fast_check_args(args)  # <- (A) required files exist?
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)

    # ffmpeg
    ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
    if osp.exists(ffmpeg_dir):
        os.environ["PATH"] += (os.pathsep + ffmpeg_dir)
    if not fast_check_ffmpeg():  # <- (B) ffmpeg installed?
        raise ImportError(
            "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script. https://ffmpeg.org/download.html")

    try:
        # feature_extract
        motion_extractor = LivePortraitMotionExtractor(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg
        )
        motion_extractor.execute(args, suffix=suffix)
    except Exception as e:
        print(f"Exception in motion extractor: {e}")
