import sqlite3


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