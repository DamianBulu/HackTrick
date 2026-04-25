import json
import sys
from pathlib import Path

sys.path.append(".")
from Tracker.tracker import Tracker2D, reassign_track_ids


def run_tracker_only(json_path: str, output_dir: str = "output", max_frames: int = 300):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {json_path} ...")
    with open(json_path) as f:
        payload = json.load(f)

    if not payload.get("success"):
        raise ValueError("JSON reports failure")

    raw_frames = payload["predictions"]["frames"]
    if max_frames and len(raw_frames) > max_frames:
        raw_frames = raw_frames[:max_frames]

    print(f"Processing {len(raw_frames)} frames.")

    # ── Fix: re-link track IDs across frames before anything else ────────────
    # The upstream detector resets track_ids every frame (each frame gets
    # brand-new IDs), so we run a lightweight IoU tracker to produce stable
    # IDs that follow each player through the clip.
    print("Re-linking track IDs across frames...")
    raw_frames = reassign_track_ids(raw_frames, iou_threshold=0.30, max_age=5)

    # ── Reconstruct mock result objects ──────────────────────────────────────
    class BoxResult:
        def __init__(self, x1, y1, x2, y2, cls_id, conf, track_id):
            self.x1       = x1
            self.y1       = y1
            self.x2       = x2
            self.y2       = y2
            self.cls_id   = cls_id
            self.conf     = conf
            self.track_id = track_id

    class FrameResult:
        def __init__(self, frame_id, boxes, keypoints):
            self.frame_id  = frame_id
            self.boxes     = boxes
            self.keypoints = keypoints

    results = []
    for raw in raw_frames:
        boxes = [
            BoxResult(
                x1=b["x1"], y1=b["y1"],
                x2=b["x2"], y2=b["y2"],
                cls_id=b["cls_id"],
                conf=b["conf"],
                track_id=b.get("track_id"),
            )
            for b in raw.get("boxes", [])
        ]
        results.append(FrameResult(
            frame_id=raw["frame_id"],
            boxes=boxes,
            keypoints=raw.get("keypoints", []),
        ))

    db_path = str(output_dir / "vid4_tracking.db")
    if Path(db_path).exists():
        Path(db_path).unlink()

    tracker = Tracker2D(db_path)

    dummy_frames = [None] * len(results)
    tracker.process_batch(results, dummy_frames)

    out_video = str(output_dir / "vid4_2d_map.mp4")
    print("Exporting 2D map video...")
    tracker.export_video(out_video, fps=25.0, trail_frames=20)
    print("Done.")


if __name__ == "__main__":
    run_tracker_only("vid4_results.json", max_frames=300)