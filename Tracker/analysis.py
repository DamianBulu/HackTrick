import sqlite3
import numpy as np
import cv2
import math
from typing import List, Tuple, Dict, Any

# Using constants from tracker if we want, but redefining the essential ones here for clarity
CLS_T1 = 6
CLS_T2 = 7

class TacticalAnalyzer:
    """
    Analyzes tactical features from the tracking database created by Tracker2D.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_positions(self, frame_id: int, cls_id: int) -> List[Tuple[float, float]]:
        """Get (pitch_x, pitch_y) for a specific class (e.g. team) in a given frame."""
        con = sqlite3.connect(self.db_path)
        rows = con.execute("""
            SELECT pitch_x, pitch_y
            FROM positions
            WHERE frame_id = ? AND cls_id = ? AND pitch_x IS NOT NULL
        """, (frame_id, cls_id)).fetchall()
        con.close()
        return rows

    def team_centroid(self, positions: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Calculates the geometric center (centroid) of the team's formation."""
        if not positions:
            return 0.0, 0.0
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        return float(np.mean(x_coords)), float(np.mean(y_coords))

    def team_length_width(self, positions: List[Tuple[float, float]]) -> Tuple[float, float]:
        """
        Calculates the length (X-axis spread) and width (Y-axis spread) of the team.
        Length represents how stretched the team is vertically.
        Width represents how stretched the team is horizontally.
        """
        if not positions:
            return 0.0, 0.0
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        return max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)

    def convex_hull_area(self, positions: List[Tuple[float, float]]) -> float:
        """
        Calculates the area of the convex hull encompassing all team players.
        This is a great metric for 'formation compactness'. A smaller area means a more compact team.
        """
        if len(positions) < 3:
            return 0.0
        pts = np.array(positions, dtype=np.float32)
        hull = cv2.convexHull(pts)
        return float(cv2.contourArea(hull))
        
    def gaps_between_lines(self, positions: List[Tuple[float, float]]) -> List[float]:
        """
        Calculates the gaps between players along the X-axis (depth).
        Returns the top 4 largest gaps, representing distances between positional lines.
        """
        if len(positions) < 2:
            return []
        x_coords = sorted([p[0] for p in positions])
        gaps = [x_coords[i+1] - x_coords[i] for i in range(len(x_coords)-1)]
        top_gaps = sorted(gaps, reverse=True)[:4]
        return [float(g) for g in top_gaps]

    def defensive_line_compactness(self, positions: List[Tuple[float, float]], centroid_x: float) -> float:
        """
        Calculates how close the defensive players are to one another.
        Identifies the 'defensive line' as the 4 deepest players, and calculates
        the average Euclidean distance between adjacent defenders in that line.
        """
        if len(positions) < 4:
            return 0.0
            
        defending_left = centroid_x < 52.5
        sorted_by_depth = sorted(positions, key=lambda p: p[0], reverse=not defending_left)
        defenders = sorted_by_depth[:4]
        
        defenders_by_width = sorted(defenders, key=lambda p: p[1])
        distances = [math.dist(defenders_by_width[i], defenders_by_width[i+1]) for i in range(len(defenders_by_width)-1)]
        return float(np.mean(distances)) if distances else 0.0

    def calculate_inter_team_distance(self, centroid1: Tuple[float, float], centroid2: Tuple[float, float]) -> float:
        """Calculates the distance between the two team centroids."""
        return math.dist(centroid1, centroid2)

    def analyze_frame(self, frame_id: int) -> Dict[str, Any]:
        """Calculate tactical metrics for a single frame."""
        t1_pos = self.get_positions(frame_id, CLS_T1)
        t2_pos = self.get_positions(frame_id, CLS_T2)
        gks = self.get_positions(frame_id, 1) # CLS_GK is 1
        
        c1 = self.team_centroid(t1_pos)
        c2 = self.team_centroid(t2_pos)

        # Assign GKs based on defending side
        gk1, gk2 = None, None
        if gks:
            if len(gks) == 1:
                # Assign to the team whose defending side is closest
                if abs(gks[0][0] - (0.0 if c1[0] < 52.5 else 105.0)) < abs(gks[0][0] - (0.0 if c2[0] < 52.5 else 105.0)):
                    gk1 = gks[0]
                else:
                    gk2 = gks[0]
            else:
                gk1 = min(gks, key=lambda gk: abs(gk[0] - (0.0 if c1[0] < 52.5 else 105.0)))
                gk2 = min(gks, key=lambda gk: abs(gk[0] - (0.0 if c2[0] < 52.5 else 105.0)))
        
        t1_with_gk = t1_pos + ([gk1] if gk1 else [])
        t2_with_gk = t2_pos + ([gk2] if gk2 else [])
        
        t1_len, t1_wid = self.team_length_width(t1_pos)
        t2_len, t2_wid = self.team_length_width(t2_pos)
        
        return {
            "frame_id": frame_id,
            "team1": {
                "player_count": len(t1_pos),
                "centroid_x": c1[0],
                "centroid_y": c1[1],
                "length": t1_len,
                "width": t1_wid,
                "convex_hull_area": self.convex_hull_area(t1_pos),
                "top_4_line_gaps": self.gaps_between_lines(t1_with_gk),
                "defender_compactness_avg_distance": self.defensive_line_compactness(t1_pos, c1[0])
            },
            "team2": {
                "player_count": len(t2_pos),
                "centroid_x": c2[0],
                "centroid_y": c2[1],
                "length": t2_len,
                "width": t2_wid,
                "convex_hull_area": self.convex_hull_area(t2_pos),
                "top_4_line_gaps": self.gaps_between_lines(t2_with_gk),
                "defender_compactness_avg_distance": self.defensive_line_compactness(t2_pos, c2[0])
            },
            "inter_team_distance": self.calculate_inter_team_distance(c1, c2)
        }

    def analyze_match(self) -> List[Dict[str, Any]]:
        """Analyzes all frames in the database."""
        con = sqlite3.connect(self.db_path)
        frame_ids = con.execute("SELECT DISTINCT frame_id FROM positions ORDER BY frame_id").fetchall()
        con.close()
        
        results = []
        for (fid,) in frame_ids:
            results.append(self.analyze_frame(fid))
        return results

if __name__ == "__main__":
    import argparse
    import json
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("db", help="Path to tracking DB")
    parser.add_argument("--output", default="output/analysis_report.json")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Error: {args.db} not found")
        exit(1)

    analyzer = TacticalAnalyzer(args.db)
    print(f"Analyzing match using {args.db}...")
    match_data = analyzer.analyze_match()
    
    with open(args.output, "w") as f:
        json.dump(match_data, f, indent=2)
    print(f"Analysis saved to {args.output}")
