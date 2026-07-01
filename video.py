"""
video.py — Pexels stock-video montage builder for Echo's /animate command.

Flow:
  1. Search Pexels for videos matching the prompt.
  2. Download up to MAX_CLIPS HD clips.
  3. Trim each clip to CLIP_DURATION seconds.
  4. Concatenate with a crossfade transition using MoviePy.
  5. Optionally overlay a free royalty-free background music track.
  6. Return the final .mp4 path (caller is responsible for cleanup).

All heavy work is synchronous and should be run via asyncio.to_thread().
"""

import logging
import os
import random
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

PEXELS_BASE = "https://api.pexels.com/videos"

# Max clips to download and stitch together.
MAX_CLIPS = 4

# Each clip is trimmed to this many seconds before stitching.
CLIP_DURATION = 4  # seconds

# Crossfade duration between clips (seconds).
CROSSFADE = 0.5

# Target output resolution — 720p is a good balance of quality vs file size.
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720

# Discord file size limit in bytes (25 MB for regular, 8 MB for non-boosted).
DISCORD_LIMIT_MB = 24


# ── Pexels helpers ────────────────────────────────────────────────────────────

def _search_pexels(query: str, api_key: str, per_page: int = 10) -> list[dict]:
    """
    Search Pexels Videos API and return a list of video result dicts.
    Falls back to a broader search if the specific query returns nothing.
    """
    params = {"query": query, "per_page": per_page, "orientation": "landscape"}
    headers = {"Authorization": api_key}

    try:
        resp = requests.get(f"{PEXELS_BASE}/search", params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        videos = resp.json().get("videos", [])
        if videos:
            return videos

        # Try popular endpoint as fallback
        log.info("Pexels returned 0 results for '%s', trying popular…", query)
        resp2 = requests.get(
            f"{PEXELS_BASE}/popular",
            params={"per_page": per_page},
            headers=headers,
            timeout=15,
        )
        resp2.raise_for_status()
        return resp2.json().get("videos", [])
    except requests.RequestException as e:
        log.error("Pexels API error: %s", e)
        return []


def _best_hd_file(video: dict) -> Optional[str]:
    """
    Pick the best HD (720p / closest) video file URL from a Pexels video dict.
    """
    files = video.get("video_files", [])
    # Sort by resolution preference: 720p first, then anything else
    hd = [f for f in files if f.get("height") == 720 and f.get("file_type") == "video/mp4"]
    sd = [f for f in files if f.get("file_type") == "video/mp4"]
    chosen = (hd or sd)
    if not chosen:
        return None
    # Take the smallest file among the chosen quality to keep download fast
    chosen.sort(key=lambda f: f.get("width", 99999))
    return chosen[0].get("link")


def _download(url: str, dest: str) -> bool:
    """Download a URL to dest path. Returns True on success."""
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
        return True
    except Exception as e:
        log.error("Download failed %s: %s", url, e)
        return False


# ── MoviePy helpers ───────────────────────────────────────────────────────────

def _build_montage(clip_paths: list[str], output_path: str) -> bool:
    """
    Trim, resize, concatenate with crossfade. Returns True on success.
    Lazily imports MoviePy so the module loads even if moviepy isn't installed yet.
    """
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        from moviepy.video.compositing.concatenate import concatenate_videoclips
        from moviepy.video.fx.resize import resize
    except ImportError:
        log.error("moviepy is not installed — cannot build montage")
        return False

    clips = []
    try:
        for path in clip_paths:
            try:
                c = VideoFileClip(path, audio=False)
                # Trim to CLIP_DURATION (or the full clip if shorter)
                end = min(CLIP_DURATION, c.duration)
                c = c.subclip(0, end)
                # Resize to target resolution maintaining aspect ratio via crop
                c = resize(c, height=OUTPUT_HEIGHT)
                # Centre-crop to OUTPUT_WIDTH
                if c.w > OUTPUT_WIDTH:
                    x_centre = c.w / 2
                    c = c.crop(
                        x1=x_centre - OUTPUT_WIDTH / 2,
                        x2=x_centre + OUTPUT_WIDTH / 2,
                    )
                clips.append(c)
            except Exception as e:
                log.warning("Skipping clip %s: %s", path, e)

        if not clips:
            return False

        if len(clips) == 1:
            final = clips[0]
        else:
            # crossfadeout on each clip except the last
            xf_clips = []
            for i, c in enumerate(clips):
                if i < len(clips) - 1:
                    xf_clips.append(c.crossfadeout(CROSSFADE))
                else:
                    xf_clips.append(c)
            final = concatenate_videoclips(xf_clips, padding=-CROSSFADE, method="compose")

        final.write_videofile(
            output_path,
            codec="libx264",
            audio=False,
            fps=24,
            preset="fast",
            logger=None,  # suppress MoviePy's verbose output
        )
        return True
    finally:
        for c in clips:
            try:
                c.close()
            except Exception:
                pass


# ── Public API ────────────────────────────────────────────────────────────────

def create_pexels_video(prompt: str, pexels_api_key: str, output_dir: str) -> Optional[str]:
    """
    Full pipeline: search → download → montage.

    Returns the path to the final .mp4, or None on failure.
    All intermediate files are written to output_dir (caller cleans up).
    """
    log.info("Pexels video for prompt: %r", prompt)

    videos = _search_pexels(prompt, pexels_api_key, per_page=10)
    if not videos:
        log.warning("No Pexels videos found for: %r", prompt)
        return None

    # Shuffle so repeated calls with the same prompt don't always grab the same clips
    random.shuffle(videos)

    downloaded: list[str] = []
    for i, video in enumerate(videos[:MAX_CLIPS * 2]):  # try up to 2× what we need
        if len(downloaded) >= MAX_CLIPS:
            break
        url = _best_hd_file(video)
        if not url:
            continue
        dest = os.path.join(output_dir, f"clip_{i}.mp4")
        if _download(url, dest):
            downloaded.append(dest)
            log.info("Downloaded clip %d/%d", len(downloaded), MAX_CLIPS)

    if not downloaded:
        return None

    output_path = os.path.join(output_dir, "montage.mp4")
    success = _build_montage(downloaded, output_path)

    if not success or not os.path.exists(output_path):
        return None

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    if size_mb > DISCORD_LIMIT_MB:
        log.warning("Output %.1f MB exceeds limit — truncating clip list and retrying", size_mb)
        # Retry with just the first clip (no montage)
        output_path2 = os.path.join(output_dir, "montage_short.mp4")
        if _build_montage(downloaded[:1], output_path2):
            return output_path2
        return None

    return output_path
