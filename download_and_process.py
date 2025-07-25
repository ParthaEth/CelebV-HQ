import os
import json
import cv2
import tqdm
from multiprocessing import Pool
import argparse
from typing import Tuple

def download(video_path, ytb_id, proxy=None):
    """
    ytb_id: youtube_id
    save_folder: save video folder
    proxy: proxy url, defalut None
    """
    if proxy is not None:
        proxy_cmd = "--proxy {}".format(proxy)
    else:
        proxy_cmd = ""
    if not os.path.exists(video_path):
        down_video = " ".join([
            "yt-dlp",
            proxy_cmd,
            "--cookies", "./assets/cookies.txt",  # ← NEW
            "-f", "'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio'",
            "--skip-unavailable-fragments",
            "--merge-output-format", "mp4",
            f"https://www.youtube.com/watch?v={ytb_id}",
            "--output", video_path,
            "--external-downloader", "aria2c",
            "--external-downloader-args", '"-x 16 -k 1M"',
            ">/dev/null 2>&1"
        ])
        # print(down_video)
        status = os.system(down_video)
        if status != 0:
            print(f"video not found: {ytb_id}")


def secs_to_timestr(secs):
    hrs = secs // 3600
    mins = (secs % 3600) // 60
    secs = secs % 60
    msec = (secs - int(secs)) * 100
    return "{:02d}:{:02d}:{:02d}.{:02d}".format(int(hrs), int(mins), int(secs), int(msec))

def expand(bbox, ratio):
    top, bottom = max(bbox[0] - ratio, 0), min(bbox[1] + ratio, 1)
    left, right = max(bbox[2] - ratio, 0), min(bbox[3] + ratio, 1)
    return top, bottom, left, right

def to_square(bbox):
    top, bottom, left, right = bbox
    h, w = bottom - top, right - left
    c = min(h, w) / 2
    c_h, c_w = (top + bottom) / 2, (left + right) / 2
    return c_h - c, c_h + c, c_w - c, c_w + c

def denorm(bbox, height, width):
    top, bottom = int(bbox[0] * height), int(bbox[1] * height)
    left, right = int(bbox[2] * width), int(bbox[3] * width)
    return top, bottom, left, right

import subprocess


def _get_dimensions(path: str) -> Tuple[int, int]:
    """
    Return (width, height) of the **first** video stream using ffprobe.
    Relies on ffmpeg‑toolchain already being on $PATH.
    """
    cmd = [
        "ffprobe",
        "-v", "error",  # suppress banner / warnings
        "-select_streams", "v:0",  # first video stream
        "-show_entries", "stream=width,height",
        "-of", "json",
        path,
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    meta = json.loads(result.stdout)
    stream = meta["streams"][0]
    return stream["width"], stream["height"]


def process_clip(args):
    """
    Crop a segment from `raw_vid_path` and save it to `save_path`,
    **without ever touching OpenCV** (avoids AV1 decoder spam).

    Args
    ----
    args : Tuple[str, str, tuple, tuple]
        raw_vid_path, save_path, bbox  (y1, y2, x1, x2 **in relative coords**),
        time  (start_sec, end_sec)

    Returns
    -------
    str : "SKIPPED" if the file already exists, else "DONE"
    """
    raw_vid_path, save_path, bbox, time = args

    # 0) Skip if work already done
    if os.path.exists(save_path):
        return "SKIPPED"

    # 1) Probe video dimensions with ffprobe (no OpenCV ≙ no AV1 noise)
    width, height = _get_dimensions(raw_vid_path)

    # 2) Convert relative bbox → absolute, make square, etc.  (your helpers)
    top, bottom, left, right = to_square(
        denorm(expand(bbox, 0.02), height, width)
    )

    start_sec, end_sec = time
    crop = f"crop=w={right - left}:h={bottom - top}:x={left}:y={top}"

    # 3) Build ffmpeg command (quiet by default)
    cmd = [
        "ffmpeg",
        "-v", "error",  # only real errors; no banners
        "-nostdin",  # don't inherit our STDIN
        "-hide_banner",
        "-i", raw_vid_path,
        "-vf", crop,
        "-ss", secs_to_timestr(start_sec),
        "-to", secs_to_timestr(end_sec),
        "-y",
        save_path,
    ]

    # 4) Run completely silenced
    with open(os.devnull, "wb") as DEVNULL:
        subprocess.run(
            cmd,
            stdout=DEVNULL,
            stderr=DEVNULL,
            stdin=subprocess.DEVNULL,
            check=True,
        )

    return "DONE"


def load_data(file_path):
    with open(file_path) as f:
        return json.load(f)


def prepare_jobs(data_dict, raw_vid_root, processed_vid_root, mode):
    jobs = []
    for key, val in data_dict['clips'].items():
        save_name = key + ".mp4"
        vid_id = val['ytb_id']
        time = val['duration']['start_sec'], val['duration']['end_sec']
        bbox = [val['bbox']['top'], val['bbox']['bottom'], val['bbox']['left'], val['bbox']['right']]

        raw_vid_path = os.path.join(raw_vid_root, vid_id + ".mp4")
        save_path = os.path.join(processed_vid_root, save_name)

        if mode == 'preprocess':
            # ✅ Only prepare job if raw video file exists
            if os.path.exists(raw_vid_path):
                jobs.append((raw_vid_path, save_path, bbox, time))
        elif mode == "download":
            # ✅ Only prepare job if raw video file DOES NOT exist
            if not os.path.exists(raw_vid_path):
                jobs.append((raw_vid_path, save_path, bbox, time))

    print(f'Total videos {len(data_dict["clips"])} and {len(jobs)} created out of that.')
    return jobs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["download", "preprocess"], required=True,
                        help="Choose whether to download raw videos or preprocess them")
    parser.add_argument("--n_process", type=int, default=2,
                        help="Choose whether to download raw videos or preprocess them")
    args = parser.parse_args()

    json_path = 'celebvhq_info.json'
    # raw_vid_root = '/home/pghosh/Downloads/dataset/CelebVHQ/raw/'
    raw_vid_root = '/home/web/data/CelebVHQ/raw/'
    # processed_vid_root = '/home/pghosh/Downloads/dataset/CelebVHQ/processed/'
    processed_vid_root = '/home/web/data/CelebVHQ/processed/'
    os.makedirs(raw_vid_root, exist_ok=True)
    os.makedirs(processed_vid_root, exist_ok=True)

    data_dict = load_data(json_path)
    jobs = prepare_jobs(data_dict, raw_vid_root, processed_vid_root, mode=args.mode)

    if args.mode == "download":
        for raw_vid_path, _, _, _ in tqdm.tqdm(jobs):
            if not os.path.exists(raw_vid_path):
                ytb_id = os.path.basename(raw_vid_path).replace(".mp4", "")
                download(raw_vid_path, ytb_id)
    elif args.mode == "preprocess":
        with Pool(processes=args.n_process) as pool:
            for result in tqdm.tqdm(pool.imap_unordered(process_clip, jobs), total=len(jobs)):
                pass