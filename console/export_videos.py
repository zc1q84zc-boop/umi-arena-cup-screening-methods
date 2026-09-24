#!/usr/bin/env python3
"""Create frame-exact, browser-playable three-view clips from LeRobot practice episodes."""
import argparse
import json
from pathlib import Path
import subprocess

from umi_arena.replay_data import Dataset

VIEWS = ("left", "right", "center")


def probe(path):
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                             "-show_entries", "stream=nb_frames,r_frame_rate,duration",
                             "-of", "json", str(path)], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)["streams"][0]


def export_clip(episode, hand, output):
    dataset = episode.dataset
    key = f"observation.image.{hand}"
    prefix = f"videos/{key}/"
    source = dataset.path(dataset.info["video_path"], video_key=key,
                          chunk_index=episode.metadata[prefix + "chunk_index"],
                          file_index=episode.metadata[prefix + "file_index"])
    start = float(episode.metadata[prefix + "from_timestamp"])
    end = float(episode.metadata[prefix + "to_timestamp"])
    if end - start + 1e-4 < episode.length / dataset.fps:
        raise ValueError(f"video window too short for episode {episode.metadata['episode_index']} {hand}")
    if output.exists() and int(probe(output).get("nb_frames", -1)) == episode.length:
        return "cached"
    temporary = output.with_suffix(".tmp.mp4")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.9f}",
           "-i", str(source), "-map", "0:v:0", "-an", "-frames:v", str(episode.length),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
           "-threads", "2", "-movflags", "+faststart", str(temporary)]
    subprocess.run(cmd, check=True)
    info = probe(temporary)
    if int(info.get("nb_frames", -1)) != episode.length or info.get("r_frame_rate") != "30/1":
        raise ValueError(f"clip mismatch: {temporary}: {info}")
    temporary.replace(output)
    return "exported"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, nargs="*")
    args = p.parse_args()
    suite = json.loads(args.suite.read_text())
    dataset = Dataset(args.dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    selected = set(args.episodes or [])
    for task in suite["tasks"]:
        for recording in task["recordings"]:
            for index in recording["episodes"]:
                if selected and index not in selected:
                    continue
                episode = dataset.episode(index)
                results = [export_clip(episode, view, args.output / f"episode-{index}-{view}.mp4") for view in VIEWS]
                print(index, episode.length, *results, flush=True)


if __name__ == "__main__":
    main()
