import sqlite3

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
SCALE_X = CANVAS_W / PITCH_W_M  # 10 px/m
SCALE_Y = CANVAS_H / PITCH_H_M  # 10 px/m

# Team colours  (BGR for OpenCV)
COLOR_TEAM1 = (60, 60, 220)  # red
COLOR_TEAM2 = (200, 80, 40)  # blue
COLOR_BALL = (30, 220, 220)  # yellow
COLOR_REF = (80, 80, 80)  # grey
COLOR_GK = (40, 200, 40)  # green

# cls_id values that come out of the agent
CLS_BALL = 0
CLS_GK = 1
CLS_REF = 3
CLS_T1 = 6
CLS_T2 = 7

# The 32 template keypoints in metres on a 105×68 pitch.
# These match TEMPLATE_F0 in agent.py scaled from the template pixel space
# (1050×680) to metres. Divide x by 10, y by 10.
TEMPLATE_METRES: List[Tuple[float, float]] = [
    (0.5, 0.5), (0.5, 14.0), (0.5, 25.0), (0.5, 43.0), (0.5, 54.0),
    (0.5, 67.5), (5.5, 25.0), (5.5, 43.0), (11.0, 34.0), (16.5, 14.0),
    (16.5, 27.0), (16.5, 41.0), (16.5, 54.0), (52.7, 0.5), (52.7, 25.3),
    (52.7, 43.3), (52.7, 67.5), (88.8, 14.0), (88.8, 27.0), (88.8, 41.0),
    (88.8, 54.0), (94.0, 34.0), (99.8, 25.0), (99.8, 43.0), (104.5, 0.5),
    (104.5, 14.0), (104.5, 25.0), (104.5, 43.0), (104.5, 54.0), (104.5, 67.5),
    (43.5, 34.0), (61.5, 34.0),
]

class Tracker2D:

    def __init__(self,db_path:str='tracking.db'):
        self.db_path = db_path
        self._last_valid_H = None
        self._h_age = 0
        self._MAX_H_CARRY_FRAMES = 15
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist yet."""
        con = sqlite3.connect(self.db_path)
        con.execute("""
                    CREATE TABLE IF NOT EXISTS positions
                    (
                        id
                        INTEGER
                        PRIMARY
                        KEY
                        AUTOINCREMENT,
                        frame_id
                        INTEGER
                        NOT
                        NULL,
                        track_id
                        INTEGER,
                        cls_id
                        INTEGER
                        NOT
                        NULL,
                        conf
                        REAL,
                        pixel_x
                        INTEGER,
                        pixel_y
                        INTEGER,
                        pitch_x
                        REAL,
                        pitch_y
                        REAL
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

    @staticmethod
    def _feet_pixel(box) -> Tuple[int, int]:
        """
        The bottom-centre of the bounding box.
        This is the point touching the ground — best input for homography.
        """
        cx = (box.x1 + box.x2) // 2
        cy = box.y2
        return cx, cy

    @staticmethod
    def _build_homography(keypoints: list) -> Optional[np.ndarray]:
        """
        Build a homography matrix H that maps frame pixel coords → pitch metres.

        keypoints: list of 32 [x, y] floats from TVFrameResult.
                   Zero pairs [0.0, 0.0] mean the keypoint wasn't detected.

        Returns H (3×3 numpy array) or None if fewer than 4 keypoints visible.
        """
        src_pts = []  # pixel positions detected in the frame
        dst_pts = []  # matching real-world positions in metres

        # There are 32 points that have been identified by the ai and we try to map it based on those
        for i, kp in enumerate(keypoints):
            x, y = float(kp[0]), float(kp[1])
            # Skip undetected keypoints
            if abs(x) < 1e-4 and abs(y) < 1e-4:
                continue
            src_pts.append([x, y])
            dst_pts.append(list(TEMPLATE_METRES[i]))

        if len(src_pts) < 4:
            return None  # not enough points to compute a homography

        src = np.array(src_pts, dtype=np.float32)
        dst = np.array(dst_pts, dtype=np.float32)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        return H

    @staticmethod
    def _project(pixel_x: int, pixel_y: int, H: np.ndarray) -> Tuple[float, float]:
        """
        Apply homography H to a single pixel point → pitch metres.
        Returns (pitch_x, pitch_y) in metres from top-left corner.
        """
        pt = np.array([[[float(pixel_x), float(pixel_y)]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, H)
        px, py = result[0][0]
        # Clamp to pitch bounds — projection can go slightly outside on edges
        px = float(np.clip(px, 0.0, PITCH_W_M))
        py = float(np.clip(py, 0.0, PITCH_H_M))
        return px, py

    def process_batch(self, results, frames):
        """
        Call this after every ai_agent.predict_batch() call.

        results: list[TVFrameResult]
        frames:  list[np.ndarray]  — the raw video frames you passed to predict_batch
        """
        rows = []

        for frame_result, frame in zip(results, frames):
            # Build homography for this frame from its keypoints
            H = self._build_homography(frame_result.keypoints)
            if H is not None:
                self._last_valid_H = H
                self._h_age = 0
            else:
                self._h_age += 1
                if self._h_age <= self._MAX_H_CARRY_FRAMES:
                    H = self._last_valid_H
                else:
                    H = None

            for box in frame_result.boxes:
                px, py = self._feet_pixel(box)

                # Project to pitch coordinates if we have a valid homography
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
        Useful for drawing movement trails.
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

        distances = {}
        prev = {}
        for track_id, frame_id, x, y in rows:
            if track_id in prev:
                px, py = prev[track_id]
                d = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
                distances[track_id] = distances.get(track_id, 0.0) + d
            prev[track_id] = (x, y)
        return distances


    @staticmethod
    def _draw_pitch(canvas: np.ndarray):
        """
        Draw standard football pitch lines on a green canvas.
        All coordinates are in canvas pixels (CANVAS_W × CANVAS_H).
        """
        W, H = CANVAS_W, CANVAS_H
        white = (255, 255, 255)
        lw = 2  # line width

        # Outer boundary
        cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), white, lw)

        # Centre line
        cv2.line(canvas, (W // 2, 0), (W // 2, H), white, lw)

        # Centre circle (radius ~9.15 m → ~91px)
        cv2.circle(canvas, (W // 2, H // 2), int(9.15 * SCALE_X), white, lw)
        cv2.circle(canvas, (W // 2, H // 2), 3, white, -1)

        # Penalty boxes (left and right)
        # Left: 16.5 m deep, 40.32 m wide centred
        pen_d = int(16.5 * SCALE_X)
        pen_w_half = int(20.16 * SCALE_Y)
        cy = H // 2
        cv2.rectangle(canvas, (0, cy - pen_w_half), (pen_d, cy + pen_w_half), white, lw)
        cv2.rectangle(canvas, (W - pen_d, cy - pen_w_half), (W, cy + pen_w_half), white, lw)

        # Goal boxes (left and right, 5.5 m deep, 18.32 m wide centred)
        gb_d = int(5.5 * SCALE_X)
        gb_w_half = int(9.16 * SCALE_Y)
        cv2.rectangle(canvas, (0, cy - gb_w_half), (gb_d, cy + gb_w_half), white, lw)
        cv2.rectangle(canvas, (W - gb_d, cy - gb_w_half), (W, cy + gb_w_half), white, lw)

        # Penalty spots
        cv2.circle(canvas, (int(11 * SCALE_X), cy), 4, white, -1)
        cv2.circle(canvas, (W - int(11 * SCALE_X), cy), 4, white, -1)

        # Corner arcs (radius 1 m)
        r = int(1.0 * SCALE_X)
        cv2.ellipse(canvas, (0, 0), (r, r), 0, 0, 90, white, lw)
        cv2.ellipse(canvas, (W, 0), (r, r), 90, 0, 90, white, lw)
        cv2.ellipse(canvas, (0, H), (r, r), 270, 0, 90, white, lw)
        cv2.ellipse(canvas, (W, H), (r, r), 180, 0, 90, white, lw)

    @staticmethod
    def _cls_to_color(cls_id: int) -> Tuple[int, int, int]:
        return {
            CLS_BALL: COLOR_BALL,
            CLS_GK: COLOR_GK,
            CLS_REF: COLOR_REF,
            CLS_T1: COLOR_TEAM1,
            CLS_T2: COLOR_TEAM2,
        }.get(cls_id, (200, 200, 200))

    @staticmethod
    def _cls_to_label(cls_id: int) -> str:
        return {
            CLS_BALL: "B",
            CLS_GK: "GK",
            CLS_REF: "R",
            CLS_T1: "1",
            CLS_T2: "2",
        }.get(cls_id, "?")

    def render_frame(
            self,
            frame_id: int,
            trail_frames: int = 20,
    ) -> np.ndarray:
        """
        Render a single 2D pitch frame.

        frame_id:     which frame to draw
        trail_frames: how many past frames to draw as a fading trail
        Returns a BGR numpy array (CANVAS_H × CANVAS_W × 3).
        """
        # Green pitch background
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        canvas[:] = (34, 100, 34)
        self._draw_pitch(canvas)

        # Draw trails (faded past positions)
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
                age = frame_id - fid  # 1 = one frame ago
                alpha = max(0.05, 1.0 - age / trail_frames)  # fade with age
                cx = int(px * SCALE_X)
                cy = int(py * SCALE_Y)
                color = tuple(int(c * alpha * 0.6) for c in self._cls_to_color(cls_id))
                cv2.circle(canvas, (cx, cy), 3, color, -1)

        # Draw current positions
        rows = self.get_frame_positions(frame_id)
        for _, track_id, cls_id, pitch_x, pitch_y in rows:
            cx = int(pitch_x * SCALE_X)
            cy = int(pitch_y * SCALE_Y)
            color = self._cls_to_color(cls_id)
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

        # Frame counter overlay
        cv2.putText(
            canvas, f"Frame {frame_id}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (200, 200, 200), 1, cv2.LINE_AA,
        )
        return canvas

    # ── Stage 7 — Video export ────────────────────────────────────────────────

    def export_video(
            self,
            output_path: str = "output_2d.mp4",
            fps: float = 25.0,
            trail_frames: int = 20,
    ):
        """
        Render every stored frame and write to a .mp4 video.

        output_path:  where to save the file
        fps:          must match your original video fps
        trail_frames: how many frames of movement trail to show
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

# ── Drop-in integration with your existing run_video.py ──────────────────────

    def run_with_2d_map(video_path: str, ai_agent, output_dir: str = "output"):
        """
        Full example: processes a video with the AI agent and produces both
        the annotated camera video AND the 2D pitch map video side by side.

        Replace your existing batch loop in run_video.py with this.
        """
        from pathlib import Path
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

        BATCH_SIZE = 64
        N_KEYPOINTS = 32

        print(f"Processing {video_path.name}  ({total_frames} frames @ {fps:.1f} fps)")

        all_frames, all_results = [], []
        num_batches = (total_frames + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_num in range(num_batches):
            start = batch_num * BATCH_SIZE
            size = min(BATCH_SIZE, total_frames - start)

            # Load frames
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

            # Save to DB immediately — no need to hold all frames in memory
            tracker.process_batch(results, frames)
            all_frames.extend(frames)
            all_results.extend(results)

        print("All batches processed. Exporting 2D map video...")
        tracker.export_video(
            str(output_dir / f"{video_path.stem}_2d_map.mp4"),
            fps=fps,
            trail_frames=20,
        )
        print("Done.")
        return tracker