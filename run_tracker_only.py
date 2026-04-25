import json
import sys
from pathlib import Path

# Add Tracker dir to path so we can import tracker
sys.path.append(".")
from Tracker.tracker import Tracker2D

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
    
    # We need to recreate the mock classes since they were deleted from tracker.py
    class BoxResult:
        def __init__(self, x1, y1, x2, y2, cls_id, conf, track_id):
            self.x1 = x1
            self.y1 = y1
            self.x2 = x2
            self.y2 = y2
            self.cls_id = cls_id
            self.conf = conf
            self.track_id = track_id

    class FrameResult:
        def __init__(self, frame_id, boxes, keypoints):
            self.frame_id = frame_id
            self.boxes = boxes
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
    # Clean up old DB if it exists
    if Path(db_path).exists():
        Path(db_path).unlink()
        
    tracker = Tracker2D(db_path)
    
    # Feed into tracker
    dummy_frames = [None] * len(results)
    tracker.process_batch(results, dummy_frames)
    
    out_video = str(output_dir / "vid4_2d_map.mp4")
    print("Exporting 2D map video...")
    tracker.export_video(out_video, fps=25.0, trail_frames=20)
    print("Done.")

if __name__ == "__main__":
    run_tracker_only("vid4_results.json", max_frames=300)
