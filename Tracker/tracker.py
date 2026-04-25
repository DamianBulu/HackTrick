import sqlite3
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

# ── Constants ─────────────────────────────────────────────────────────────────

# Standard pitch size in metres (UEFA regulation)
PITCH_W_M = 105.0
PITCH_H_M = 68.0

# Canvas size for the 2D render (pixels)
CANVAS_W = 1050
CANVAS_H = 680

# Scale: how many pixels per metre
SCALE_X = CANVAS_W / PITCH_W_M   # 10 px/m
SCALE_Y = CANVAS_H / PITCH_H_M   # 10 px/m

# Team colours  (BGR for OpenCV)
COLOR_TEAM1 = (60,  60,  220)   # red
COLOR_TEAM2 = (200, 80,  40)    # blue
COLOR_BALL  = (30,  220, 220)   # yellow
COLOR_REF   = (80,  80,  80)    # grey
COLOR_GK    = (40,  200, 40)    # green

# cls_id values that come out of the agent
CLS_BALL = 0
CLS_GK   = 1
CLS_REF  = 3
CLS_T1   = 6
CLS_T2   = 7

# The 32 template keypoints in metres on a 105×68 pitch.
TEMPLATE_METRES: List[Tuple[float, float]] = [
    (0.5,  0.5),  (0.5,  14.0), (0.5,  25.0), (0.5,  43.0), (0.5,  54.0),
    (0.5,  67.5), (5.5,  25.0), (5.5,  43.0), (11.0, 34.0), (16.5, 14.0),
    (16.5, 27.0), (16.5, 41.0), (16.5, 54.0), (52.7, 0.5),  (52.7, 25.3),
    (52.7, 43.3), (52.7, 67.5), (88.8, 14.0), (88.8, 27.0), (88.8, 41.0),
    (88.8, 54.0), (94.0, 34.0), (99.8, 25.0), (99.8, 43.0), (104.5, 0.5),
    (104.5, 14.0),(104.5, 25.0),(104.5, 43.0),(104.5, 54.0),(104.5, 67.5),
    (43.5, 34.0), (61.5, 34.0),
]

# ── Main tracker ──────────────────────────────────────────────────────────────

class Tracker2D:

    def __init__(self, db_path: str = "tracking.db"):
        self.db_path = db_path
        self._last_valid_H: Optional[np.ndarray] = None
        self._h_age    = 0
        # Increased from 15 → 40 because keypoints are sparse and unreliable;
        # carrying a stale-but-good H is safer than projecting with a bad one.
        self._MAX_H_CARRY_FRAMES = 40
        # EMA weight for blending new homography with previous (0 = full carry,
        # 1 = no smoothing).  0.7 damps frame-to-frame flicker nicely.
        self._H_EMA_ALPHA = 0.7
        self._init_db()

    # ── Database ──────────────────────────────────────────────────────────────

    def _init_db(self):
        """Create tables if they don't exist yet."""
        con = sqlite3.connect(self.db_path)
        con.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id INTEGER NOT NULL,
                track_id INTEGER,
                cls_id   INTEGER NOT NULL,
                conf     REAL,
                pixel_x  INTEGER,
                pixel_y  INTEGER,
                pitch_x  REAL,
                pitch_y  REAL
            )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_frame ON positions(frame_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_track ON positions(track_id)")
        con.commit()
        con.close()

    def _insert_rows(self, rows: list):
        con = sqlite3.connect(self.db_path)
        con.executemany("""
            INSERT INTO positions
                (frame_id, track_id, cls_id, conf, pixel_x, pixel_y, pitch_x, pitch_y)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        con.commit()
        con.close()

    # ── Geometry helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _feet_pixel(box) -> Tuple[int, int]:
        """
        The bottom-centre of the bounding box.
        Fixed: use proper int() cast instead of floor-division on floats.
        """
        cx = int((box.x1 + box.x2) / 2)
        cy = int(box.y2)
        return cx, cy

    @staticmethod
    def _homography_is_sane(H: np.ndarray) -> bool:
        """
        Basic check: H must be non-None and numerically finite.
        A geometric bounds check is not reliable when the camera shows only
        a partial pitch (corners outside the frame legitimately project far
        outside pitch bounds). RANSAC already handles outlier rejection.
        """
        if H is None:
            return False
        return bool(np.all(np.isfinite(H)))

    @staticmethod
    def _build_homography(keypoints: list) -> Optional[np.ndarray]:
        """
        Build a homography matrix H that maps frame pixel coords → pitch metres.

        keypoints: list of 32 [x, y] floats.
                   Zero pairs [0.0, 0.0] mean the keypoint wasn't detected.

        Changes vs original:
        - RANSAC reprojection threshold tightened from 5.0 → 3.0 px
        - Returns None if fewer than 6 (was 4) keypoints visible, for stability
        - Sanity-checked by caller before it is accepted
        """
        src_pts, dst_pts = [], []
        for i, kp in enumerate(keypoints):
            x, y = float(kp[0]), float(kp[1])
            if abs(x) < 1e-4 and abs(y) < 1e-4:
                continue
            src_pts.append([x, y])
            dst_pts.append(list(TEMPLATE_METRES[i]))

        # Raised minimum from 4 → 6 for better-conditioned homography
        if len(src_pts) < 6:
            return None

        src = np.array(src_pts, dtype=np.float32)
        dst = np.array(dst_pts, dtype=np.float32)
        # Tightened reprojection threshold: 5.0 → 3.0 px
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        return H

    @staticmethod
    def _project(pixel_x: int, pixel_y: int, H: np.ndarray) -> Tuple[float, float]:
        """
        Apply homography H to a single pixel point → pitch metres.
        Returns (pitch_x, pitch_y) clamped to pitch bounds.
        """
        pt = np.array([[[float(pixel_x), float(pixel_y)]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, H)
        px, py = result[0][0]
        px = float(np.clip(px, 0.0, PITCH_W_M))
        py = float(np.clip(py, 0.0, PITCH_H_M))
        return px, py

    # ── Core processing ───────────────────────────────────────────────────────

    def process_batch(self, results, frames):
        """
        Call this after every ai_agent.predict_batch() call.

        results : list[TVFrameResult]
        frames  : list[np.ndarray]  — raw video frames (may be list of None)

        Key changes vs original:
        - New homography is sanity-checked before being accepted
        - EMA blending of H instead of hard carry-forward to reduce flicker
        - Carry limit raised to 40 frames
        """
        rows = []

        for frame_result, frame in zip(results, frames):
            H_new = self._build_homography(frame_result.keypoints)

            if H_new is not None and self._homography_is_sane(H_new):
                if self._last_valid_H is not None:
                    # EMA blend: smooth transition rather than hard switch
                    H = self._H_EMA_ALPHA * H_new + (1 - self._H_EMA_ALPHA) * self._last_valid_H
                else:
                    H = H_new
                self._last_valid_H = H
                self._h_age = 0
            else:
                self._h_age += 1
                if self._h_age <= self._MAX_H_CARRY_FRAMES:
                    H = self._last_valid_H
                else:
                    H = None   # too stale; skip projection this frame

            for box in frame_result.boxes:
                px, py = self._feet_pixel(box)

                pitch_x, pitch_y = None, None
                if H is not None:
                    pitch_x, pitch_y = self._project(px, py, H)

                rows.append((
                    frame_result.frame_id,
                    box.track_id,
                    box.cls_id,
                    round(box.conf, 4),
                    px, py,
                    pitch_x, pitch_y,
                ))

        self._insert_rows(rows)

    # ── Query helpers ─────────────────────────────────────────────────────────

    def get_frame_positions(self, frame_id: int) -> list:
        """All detected objects at a single frame."""
        con = sqlite3.connect(self.db_path)
        rows = con.execute("""
            SELECT frame_id, track_id, cls_id, pitch_x, pitch_y
            FROM positions
            WHERE frame_id = ?
              AND pitch_x IS NOT NULL
        """, (frame_id,)).fetchall()
        con.close()
        return rows

    def get_all_frame_ids(self) -> List[int]:
        """Sorted list of every frame_id stored in the DB."""
        con = sqlite3.connect(self.db_path)
        rows = con.execute(
            "SELECT DISTINCT frame_id FROM positions ORDER BY frame_id"
        ).fetchall()
        con.close()
        return [r[0] for r in rows]

    def get_trajectory(self, track_id: int) -> List[Tuple[int, float, float]]:
        """
        (frame_id, pitch_x, pitch_y) for one player across the whole match.
        Requires stable track_ids (use reassign_track_ids() if upstream
        detector resets them every frame).
        """
        con = sqlite3.connect(self.db_path)
        rows = con.execute("""
            SELECT frame_id, pitch_x, pitch_y
            FROM positions
            WHERE track_id = ?
              AND pitch_x IS NOT NULL
            ORDER BY frame_id
        """, (track_id,)).fetchall()
        con.close()
        return rows

    def get_distance_covered(self) -> dict:
        """
        Total metres run per track_id across all frames.
        Returns {track_id: metres_float}
        """
        con = sqlite3.connect(self.db_path)
        rows = con.execute("""
            SELECT track_id, frame_id, pitch_x, pitch_y
            FROM positions
            WHERE track_id IS NOT NULL
              AND pitch_x IS NOT NULL
            ORDER BY track_id, frame_id
        """).fetchall()
        con.close()

        distances: dict = {}
        prev: dict = {}
        for track_id, frame_id, x, y in rows:
            if track_id in prev:
                px, py = prev[track_id]
                d = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
                distances[track_id] = distances.get(track_id, 0.0) + d
            prev[track_id] = (x, y)
        return distances

    # ── Rendering ─────────────────────────────────────────────────────────────

    @staticmethod
    def _draw_pitch(canvas: np.ndarray):
        """Draw standard football pitch lines on a green canvas."""
        W, H   = CANVAS_W, CANVAS_H
        white  = (255, 255, 255)
        lw     = 2

        cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), white, lw)
        cv2.line(canvas, (W // 2, 0), (W // 2, H), white, lw)
        cv2.circle(canvas, (W // 2, H // 2), int(9.15 * SCALE_X), white, lw)
        cv2.circle(canvas, (W // 2, H // 2), 3, white, -1)

        pen_d      = int(16.5 * SCALE_X)
        pen_w_half = int(20.16 * SCALE_Y)
        cy         = H // 2
        cv2.rectangle(canvas, (0, cy - pen_w_half), (pen_d,     cy + pen_w_half), white, lw)
        cv2.rectangle(canvas, (W - pen_d, cy - pen_w_half), (W, cy + pen_w_half), white, lw)

        gb_d      = int(5.5  * SCALE_X)
        gb_w_half = int(9.16 * SCALE_Y)
        cv2.rectangle(canvas, (0, cy - gb_w_half), (gb_d,     cy + gb_w_half), white, lw)
        cv2.rectangle(canvas, (W - gb_d, cy - gb_w_half), (W, cy + gb_w_half), white, lw)

        cv2.circle(canvas, (int(11 * SCALE_X),     cy), 4, white, -1)
        cv2.circle(canvas, (W - int(11 * SCALE_X), cy), 4, white, -1)

        r = int(1.0 * SCALE_X)
        cv2.ellipse(canvas, (0, 0), (r, r), 0,   0, 90,  white, lw)
        cv2.ellipse(canvas, (W, 0), (r, r), 90,  0, 90,  white, lw)
        cv2.ellipse(canvas, (0, H), (r, r), 270, 0, 90,  white, lw)
        cv2.ellipse(canvas, (W, H), (r, r), 180, 0, 90,  white, lw)

    @staticmethod
    def _cls_to_color(cls_id: int) -> Tuple[int, int, int]:
        return {
            CLS_BALL: COLOR_BALL,
            CLS_GK:   COLOR_GK,
            CLS_REF:  COLOR_REF,
            CLS_T1:   COLOR_TEAM1,
            CLS_T2:   COLOR_TEAM2,
        }.get(cls_id, (200, 200, 200))

    @staticmethod
    def _cls_to_label(cls_id: int) -> str:
        return {
            CLS_BALL: "B",
            CLS_GK:   "GK",
            CLS_REF:  "R",
            CLS_T1:   "1",
            CLS_T2:   "2",
        }.get(cls_id, "?")

    def render_frame(
        self,
        frame_id: int,
        trail_frames: int = 20,
    ) -> np.ndarray:
        """
        Render a single 2D pitch frame.

        frame_id     : which frame to draw
        trail_frames : how many past frames to draw as a fading trail
        Returns a BGR numpy array (CANVAS_H × CANVAS_W × 3).
        """
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        canvas[:] = (34, 100, 34)
        self._draw_pitch(canvas)

        if trail_frames > 0:
            con = sqlite3.connect(self.db_path)
            trail_rows = con.execute("""
                SELECT track_id, cls_id, pitch_x, pitch_y, frame_id
                FROM positions
                WHERE frame_id BETWEEN ? AND ?
                  AND pitch_x IS NOT NULL
                  AND track_id IS NOT NULL
                ORDER BY frame_id
            """, (frame_id - trail_frames, frame_id - 1)).fetchall()
            con.close()

            for tid, cls_id, px, py, fid in trail_rows:
                age   = frame_id - fid
                alpha = max(0.05, 1.0 - age / trail_frames)
                cx    = int(px * SCALE_X)
                cy    = int(py * SCALE_Y)
                color = tuple(int(c * alpha * 0.6) for c in self._cls_to_color(cls_id))
                cv2.circle(canvas, (cx, cy), 3, color, -1)

        rows = self.get_frame_positions(frame_id)
        for _, track_id, cls_id, pitch_x, pitch_y in rows:
            cx     = int(pitch_x * SCALE_X)
            cy     = int(pitch_y * SCALE_Y)
            color  = self._cls_to_color(cls_id)
            radius = 10 if cls_id == CLS_BALL else 14

            cv2.circle(canvas, (cx, cy), radius, color, -1)
            cv2.circle(canvas, (cx, cy), radius, (255, 255, 255), 1)

            label = self._cls_to_label(cls_id)
            if track_id:
                label = str(track_id)

            cv2.putText(
                canvas, label,
                (cx - 5, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

        cv2.putText(
            canvas, f"Frame {frame_id}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (200, 200, 200), 1, cv2.LINE_AA,
        )
        return canvas

    # ── Video export ──────────────────────────────────────────────────────────

    def export_video(
        self,
        output_path: str = "output_2d.mp4",
        fps: float = 25.0,
        trail_frames: int = 20,
    ):
        """
        Render every stored frame and write to a .mp4 video.

        output_path  : where to save the file
        fps          : must match your original video fps
        trail_frames : how many frames of movement trail to show
        """
        frame_ids = self.get_all_frame_ids()
        if not frame_ids:
            print("No frames in database yet.")
            return

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (CANVAS_W, CANVAS_H))

        print(f"Exporting {len(frame_ids)} frames → {output_path}")
        for i, fid in enumerate(frame_ids):
            frame = self.render_frame(fid, trail_frames=trail_frames)
            writer.write(frame)
            if i % 100 == 0:
                print(f"  {i}/{len(frame_ids)} frames done")

        writer.release()
        print(f"Done. Saved to {output_path}")

    # ── Drop-in integration ───────────────────────────────────────────────────

    @staticmethod
    def run_with_2d_map(video_path: str, ai_agent, output_dir: str = "output"):
        """
        Full example: processes a video with the AI agent and produces both
        the annotated camera video AND the 2D pitch map video.
        """
        import cv2 as _cv2

        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        db_path = str(output_dir / f"{video_path.stem}_tracking.db")
        tracker = Tracker2D(db_path)

        cap = _cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(_cv2.CAP_PROP_FPS)
        cap.release()

        BATCH_SIZE  = 64
        N_KEYPOINTS = 32

        print(f"Processing {video_path.name}  ({total_frames} frames @ {fps:.1f} fps)")

        num_batches = (total_frames + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_num in range(num_batches):
            start = batch_num * BATCH_SIZE
            size  = min(BATCH_SIZE, total_frames - start)

            cap = _cv2.VideoCapture(str(video_path))
            cap.set(_cv2.CAP_PROP_POS_FRAMES, start)
            frames = []
            for _ in range(size):
                ret, f = cap.read()
                if not ret:
                    break
                frames.append(f)
            cap.release()

            if not frames:
                break

            print(f"  Batch {batch_num + 1}/{num_batches}")
            results = ai_agent.predict_batch(frames, offset=start, n_keypoints=N_KEYPOINTS)
            tracker.process_batch(results, frames)

        print("All batches processed. Exporting 2D map video...")
        tracker.export_video(
            str(output_dir / f"{video_path.stem}_2d_map.mp4"),
            fps=fps,
            trail_frames=20,
        )
        print("Done.")
        return tracker


# ── JSON pipeline (run directly from CLI) ─────────────────────────────────────

class _BoxResult:
    """Minimal stand-in for TVBoxResult when loading from a JSON payload."""
    __slots__ = ("x1", "y1", "x2", "y2", "cls_id", "conf", "track_id")

    def __init__(self, x1, y1, x2, y2, cls_id, conf, track_id):
        self.x1       = x1
        self.y1       = y1
        self.x2       = x2
        self.y2       = y2
        self.cls_id   = cls_id
        self.conf     = conf
        self.track_id = track_id


class _FrameResult:
    """Minimal stand-in for TVFrameResult when loading from a JSON payload."""
    __slots__ = ("frame_id", "boxes", "keypoints")

    def __init__(self, frame_id, boxes, keypoints):
        self.frame_id  = frame_id
        self.boxes     = boxes
        self.keypoints = keypoints


def process_json(
    json_path: str,
    output_dir: str   = "output",
    max_frames: int   = 0,
    fps: float        = 25.0,
    trail_frames: int = 20,
) -> "Tracker2D":
    """
    Load a prediction JSON, run the 2-D tracker and export a pitch-map video.

    Parameters
    ----------
    json_path     : path to the JSON file produced by the AI agent
    output_dir    : folder where the .db and .mp4 are written
    max_frames    : cap on frames to process (0 = all)
    fps           : frame-rate for the output video
    trail_frames  : length of movement trail drawn on each frame

    Returns
    -------
    The populated Tracker2D instance (DB already written).
    """
    import json as _json

    json_path  = Path(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"[1/4] Loading {json_path.name} ...")
    with open(json_path) as f:
        payload = _json.load(f)

    if not payload.get("success"):
        raise ValueError("JSON payload reports success=false")

    raw_frames = payload["predictions"]["frames"]
    total = len(raw_frames)
    if max_frames and total > max_frames:
        raw_frames = raw_frames[:max_frames]
    print(f"      {len(raw_frames)} / {total} frames selected.")

    # ── 2. Build result objects ───────────────────────────────────────────────
    print("[2/4] Building frame objects ...")
    results = []
    for raw in raw_frames:
        boxes = [
            _BoxResult(
                x1=b["x1"], y1=b["y1"], x2=b["x2"], y2=b["y2"],
                cls_id=b["cls_id"], conf=b["conf"],
                track_id=b.get("track_id"),
            )
            for b in raw.get("boxes", [])
        ]
        results.append(_FrameResult(
            frame_id  = raw["frame_id"],
            boxes     = boxes,
            keypoints = raw.get("keypoints", []),
        ))

    # ── 3. Track ──────────────────────────────────────────────────────────────
    db_path = str(output_dir / f"{json_path.stem}_tracking.db")
    if Path(db_path).exists():
        Path(db_path).unlink()

    print(f"[3/4] Running 2-D tracker → {Path(db_path).name} ...")
    tracker = Tracker2D(db_path)
    tracker.process_batch(results, [None] * len(results))

    # ── 4. Export video ───────────────────────────────────────────────────────
    out_video = str(output_dir / f"{json_path.stem}_2d_map.mp4")
    print(f"[4/4] Exporting video → {Path(out_video).name} ...")
    tracker.export_video(out_video, fps=fps, trail_frames=trail_frames)

    return tracker


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process an AI-agent prediction JSON and produce a 2-D pitch-map video.",
    )
    parser.add_argument("json",            help="Path to the prediction JSON file")
    parser.add_argument("-o", "--output",  default="output",  help="Output directory (default: output)")
    parser.add_argument("-n", "--frames",  default=0, type=int, help="Max frames to process (default: all)")
    parser.add_argument("--fps",           default=25.0, type=float, help="Output video FPS (default: 25)")
    parser.add_argument("--trail",         default=20,   type=int,   help="Trail length in frames (default: 20)")
    args = parser.parse_args()

    process_json(
        json_path    = args.json,
        output_dir   = args.output,
        max_frames   = args.frames,
        fps          = args.fps,
        trail_frames = args.trail,
    )