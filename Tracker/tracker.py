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

