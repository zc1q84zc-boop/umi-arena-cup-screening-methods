#!/usr/bin/env python3
"""Read-only dashboard for UMI Arena Track 1 practice recordings and replay results."""
import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
VIDEOS = ROOT / "videos"
VIDEO_NAME = re.compile(r"^/videos/episode-(\d+)-(left|right|center)\.mp4$")
CHECKPOINTS = ROOT / "checkpoints.json"


def read_json(path):
    return json.loads(path.read_text())


def checkpoint_catalog(replay_root, revision):
    entries = read_json(CHECKPOINTS)["checkpoints"]
    if len({entry["id"] for entry in entries}) != len(entries):
        raise ValueError("duplicate checkpoint id")
    result = []
    for entry in entries:
        report_path = replay_root / entry["id"] / "report.json"
        report = read_json(report_path) if report_path.is_file() else None
        valid = bool(report and report.get("checkpoint_id") == entry["id"]
                     and report.get("dataset_revision") == revision and report.get("baseline") is None)
        completed = sum(e.get("status") == "complete" for e in report.get("episodes", [])) if valid else 0
        result.append({"id": entry["id"], "step": entry["step"], "label": entry["label"],
                       "task_id": entry["task_id"], "completed_episodes": completed,
                       "replay_status": report.get("status", "pending") if valid else "pending"})
    return result


def prediction_for(episode_id, replay_dir, revision, expected_checkpoint_id=None):
    report_path = replay_dir / "report.json"
    if not report_path.is_file():
        return {"available": False, "reason": "checkpoint 已登记；该片段的模型预测尚未生成。"}
    report = read_json(report_path)
    if report.get("baseline") is not None or not report.get("checkpoint_id"):
        return {"available": False, "reason": "当前 replay 是 baseline，不是已训练模型预测。"}
    if expected_checkpoint_id and report["checkpoint_id"] != expected_checkpoint_id:
        return {"available": False, "reason": "replay 的 checkpoint ID 与所选模型不一致。"}
    if report.get("dataset_revision") != revision:
        return {"available": False, "reason": "replay 与 practice 数据集版本不一致。"}
    matches = [e for e in report.get("episodes", []) if e.get("episode_index") == episode_id and e.get("status") == "complete"]
    if not matches:
        reason = ("该片段已从清洁数据中隔离，未运行模型预测。"
                  if episode_id in report.get("excluded_practice_episodes", []) else
                  "该片段没有完成的模型预测。")
        return {"available": False, "reason": reason, "checkpoint_id": report["checkpoint_id"]}
    entry = sorted(matches, key=lambda e: e.get("repeat", 0))[0]
    if not entry.get("windows"):
        detail = replay_dir / "episodes" / f"episode-{episode_id}-repeat-{entry.get('repeat', 0)}.json"
        if detail.is_file():
            entry = read_json(detail)
    windows = []
    timing = report.get("settings", {}).get("pose_timing", "previous")
    for w in entry.get("windows", []):
        offset = 0 if timing == "previous" else 1
        windows.append({"frame": w["frame"], "indices": [w["frame"] + offset + i for i in range(len(w["predicted_poses"]))],
                        "poses": w["predicted_poses"], "grippers": w["predicted_grippers"],
                        "position_cm": w.get("position_cm"), "rotation_deg": w.get("rotation_deg")})
    return {"available": bool(windows), "checkpoint_id": report["checkpoint_id"],
            "repeat": entry.get("repeat", 0), "windows": windows,
            "summary": entry.get("summary", {}), "status": report.get("status"),
            "training_overlap": entry.get("training_overlap", False)}


def handler_factory(replay_dir):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(ROOT), **kwargs)

        def do_GET(self):
            url = urlsplit(self.path)
            if VIDEO_NAME.fullmatch(url.path):
                return self.send_video(url.path, include_body=True)
            if url.path == "/api/catalog":
                catalog = read_json(DATA / "catalog.json")
                return self.send_data({**catalog, "checkpoints": checkpoint_catalog(replay_dir, catalog["dataset_revision"])})
            if url.path == "/api/episode":
                try:
                    query = parse_qs(url.query)
                    episode_id = int(query["id"][0])
                    catalog = read_json(DATA / "catalog.json")
                    matching = [e for e in catalog["episodes"] if e["episode_index"] == episode_id]
                    if not matching:
                        raise ValueError("episode is outside the practice suite")
                    checkpoints = {e["id"]: e for e in checkpoint_catalog(replay_dir, catalog["dataset_revision"])}
                    checkpoint_id = query.get("checkpoint", [max(checkpoints.values(), key=lambda e: e["step"])["id"]])[0]
                    if checkpoint_id not in checkpoints:
                        raise ValueError("unknown checkpoint")
                    episode = read_json(DATA / f"episode-{episode_id}.json")
                    episode["videos"] = {hand: f"/videos/episode-{episode_id}-{hand}.mp4"
                                         if (VIDEOS / f"episode-{episode_id}-{hand}.mp4").is_file() else None
                                         for hand in ("left", "right", "center")}
                    selected = checkpoints[checkpoint_id]
                    prediction = (prediction_for(episode_id, replay_dir / checkpoint_id,
                                                  catalog["dataset_revision"], checkpoint_id)
                                  if matching[0]["task_id"] == selected["task_id"] else
                                  {"available": False, "reason": "此 checkpoint 只训练杯子任务；手机任务未生成模型预测。"})
                    return self.send_data({"recording": episode, "checkpoint": selected,
                                           "prediction": prediction})
                except (KeyError, ValueError, IndexError, OSError) as exc:
                    return self.send_data({"error": str(exc)}, 400)
            if url.path.startswith(("/data/", "/replay_results/", "/videos/")):
                return self.send_error(404)
            return super().do_GET()

        def do_HEAD(self):
            url = urlsplit(self.path)
            if VIDEO_NAME.fullmatch(url.path):
                return self.send_video(url.path, include_body=False)
            return super().do_HEAD()

        def send_video(self, url_path, include_body):
            match = VIDEO_NAME.fullmatch(url_path)
            path = VIDEOS / f"episode-{match.group(1)}-{match.group(2)}.mp4"
            if not path.is_file():
                return self.send_error(404)
            size = path.stat().st_size
            start, end, status = 0, size - 1, 200
            requested = self.headers.get("Range", "")
            if requested:
                parsed = re.fullmatch(r"bytes=(\d+)-(\d*)", requested)
                if not parsed:
                    return self.send_error(416)
                start = int(parsed.group(1))
                end = min(int(parsed.group(2)), size - 1) if parsed.group(2) else size - 1
                if start >= size or end < start:
                    return self.send_error(416)
                status = 206
            self.send_response(status)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "public, max-age=3600")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if include_body:
                with path.open("rb") as video:
                    video.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        block = video.read(min(256 * 1024, remaining))
                        if not block:
                            break
                        try:
                            self.wfile.write(block)
                        except (BrokenPipeError, ConnectionResetError):
                            break
                        remaining -= len(block)

        def send_data(self, payload, status=200):
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8768)
    p.add_argument("--replay", type=Path, default=ROOT / "replay_results", help="directory containing one replay report subdirectory per registered checkpoint")
    args = p.parse_args()
    if not (DATA / "catalog.json").is_file():
        p.error("practice data missing; run export_practice.py and copy its output into data/")
    server = ThreadingHTTPServer((args.host, args.port), handler_factory(args.replay.resolve()))
    print(f"Track 1 console: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
