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
        self.init_db()

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