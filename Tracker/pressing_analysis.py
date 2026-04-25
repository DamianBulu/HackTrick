import sqlite3
import math
import numpy as np
import os
from typing import List, Tuple, Dict, Any, Optional

CLS_BALL = 0
CLS_T1 = 6
CLS_T2 = 7
CLS_GK = 1

PITCH_W = 105.0
PITCH_H = 68.0

class PressingAnalyzer:
    """
    Analyzes pressing statistics based on the tracking database.
    It applies a sliding window to smooth possession and classify pressing styles.
    It groups windows into discrete 'Pressing Events' to measure effectiveness.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_positions(self, frame_id: int, cls_id: int) -> List[Tuple[float, float]]:
        con = sqlite3.connect(self.db_path)
        rows = con.execute("""
            SELECT pitch_x, pitch_y
            FROM positions
            WHERE frame_id = ? AND cls_id = ? AND pitch_x IS NOT NULL
        """, (frame_id, cls_id)).fetchall()
        con.close()
        return rows

    def get_ball_position(self, frame_id: int) -> Optional[Tuple[float, float]]:
        con = sqlite3.connect(self.db_path)
        row = con.execute("""
            SELECT pitch_x, pitch_y
            FROM positions
            WHERE frame_id = ? AND cls_id = ? AND pitch_x IS NOT NULL
            LIMIT 1
        """, (frame_id, CLS_BALL)).fetchone()
        con.close()
        
        if row: return row
        return self._approximate_ball_position(frame_id)

    def _approximate_ball_position(self, frame_id: int) -> Optional[Tuple[float, float]]:
        t1 = self.get_positions(frame_id, CLS_T1)
        t2 = self.get_positions(frame_id, CLS_T2)
        all_players = t1 + t2
        if not all_players:
            return None
            
        best_player = None
        min_dist = float('inf')
        for p in all_players:
            dists = sorted([math.dist(p, other) for other in all_players if other != p])
            if len(dists) >= 3:
                score = sum(dists[:3])
                if score < min_dist:
                    min_dist = score
                    best_player = p
        return best_player if best_player else all_players[0]

    def get_team_defending_side(self, frame_id: int, team_cls: int) -> float:
        positions = self.get_positions(frame_id, team_cls)
        if not positions:
            return 0.0
        centroid_x = np.mean([p[0] for p in positions])
        return 0.0 if centroid_x < (PITCH_W / 2) else PITCH_W

    def get_pitch_zone(self, x: float, y: float) -> str:
        third_x = "Defensive" if x < 35.0 else ("Middle" if x < 70.0 else "Attacking")
        flank_y = "Left" if y < 22.6 else ("Center" if y < 45.3 else "Right")
        return f"{third_x}_{flank_y}"

    def analyze_frame(self, frame_id: int) -> Dict[str, Any]:
        ball = self.get_ball_position(frame_id)
        if not ball:
            return {"frame_id": frame_id, "pressing_metrics": None}

        t1_pos = self.get_positions(frame_id, CLS_T1)
        t2_pos = self.get_positions(frame_id, CLS_T2)

        if not t1_pos and not t2_pos:
            return {"frame_id": frame_id, "pressing_metrics": None}

        t1_dists = sorted([math.dist(p, ball) for p in t1_pos]) if t1_pos else []
        t2_dists = sorted([math.dist(p, ball) for p in t2_pos]) if t2_pos else []

        min_t1 = t1_dists[0] if t1_dists else float('inf')
        min_t2 = t2_dists[0] if t2_dists else float('inf')

        possessing_team = "Team 1" if min_t1 < min_t2 else "Team 2"
        defending_team = "Team 2" if possessing_team == "Team 1" else "Team 1"
        defending_dists = t2_dists if possessing_team == "Team 1" else t1_dists

        if not defending_dists:
            return {"frame_id": frame_id, "pressing_metrics": None}

        closest_defender_dist = defending_dists[0]
        players_in_5m = len([d for d in defending_dists if d <= 5.0])
        players_in_10m = len([d for d in defending_dists if d <= 10.0])
        closest_3_avg_dist = float(np.mean(defending_dists[:3])) if len(defending_dists) >= 3 else float(np.mean(defending_dists))

        is_active_press = closest_defender_dist < 4.0

        defending_side = self.get_team_defending_side(frame_id, CLS_T2 if defending_team == "Team 2" else CLS_T1)
        is_high_press = False
        if is_active_press:
            if defending_side == 0.0 and ball[0] > 70.0:
                is_high_press = True
            elif defending_side == PITCH_W and ball[0] < 35.0:
                is_high_press = True

        return {
            "frame_id": frame_id,
            "ball_position": ball,
            "possessing_team": possessing_team,
            "pressing_metrics": {
                "pressing_team": defending_team,
                "closest_defender_distance": closest_defender_dist,
                "players_within_5m": players_in_5m,
                "players_within_10m": players_in_10m,
                "avg_distance_closest_3_defenders": closest_3_avg_dist,
                "is_active_press": is_active_press,
                "is_high_press": is_high_press,
                "pressing_zone": self.get_pitch_zone(ball[0], ball[1]) if is_active_press else None
            }
        }

    def analyze_match(self, window_size: int = 3) -> Dict[str, Any]:
        con = sqlite3.connect(self.db_path)
        frame_ids = con.execute("SELECT DISTINCT frame_id FROM positions ORDER BY frame_id").fetchall()
        con.close()
        
        raw_frames = []
        for (fid,) in frame_ids:
            res = self.analyze_frame(fid)
            if res.get("pressing_metrics") is not None:
                raw_frames.append(res)
                
        # ── Apply Sliding Window for Smoothing ──
        smoothed_windows = []
        for i in range(len(raw_frames) - window_size + 1):
            window = raw_frames[i:i + window_size]
            
            possessing_teams = [f["possessing_team"] for f in window]
            majority_possessing = max(set(possessing_teams), key=possessing_teams.count)
            defending_team = "Team 2" if majority_possessing == "Team 1" else "Team 1"
            
            valid_metrics = [f["pressing_metrics"] for f in window if f["possessing_team"] == majority_possessing]
            if not valid_metrics:
                continue
                
            avg_in_5m = sum(m["players_within_5m"] for m in valid_metrics) / len(valid_metrics)
            avg_in_10m = sum(m["players_within_10m"] for m in valid_metrics) / len(valid_metrics)
            
            active_press_count = sum(1 for m in valid_metrics if m["is_active_press"])
            high_press_count = sum(1 for m in valid_metrics if m["is_high_press"])
            
            is_active_press = active_press_count >= (len(valid_metrics) / 2.0)
            is_high_press = high_press_count >= (len(valid_metrics) / 2.0)
            
            pressing_style = "No Press"
            if is_active_press:
                if avg_in_10m >= 3.0:
                    pressing_style = "Organized Press"
                elif avg_in_10m >= 2.0:
                    pressing_style = "Supported Press"
                else:
                    pressing_style = "Single Player Press"
            
            zones = [m["pressing_zone"] for m in valid_metrics if m["pressing_zone"]]
            majority_zone = max(set(zones), key=zones.count) if zones else None
            
            smoothed_windows.append({
                "window": f"F{window[0]['frame_id']}-F{window[-1]['frame_id']}",
                "possessing_team": majority_possessing,
                "defending_team": defending_team,
                "is_active_press": is_active_press,
                "is_high_press": is_high_press,
                "pressing_style": pressing_style,
                "pressing_zone": majority_zone
            })

        # ── Group into Pressing Sequences/Events ──
        pressing_events = []
        if smoothed_windows:
            style_hierarchy = {"No Press": 0, "Single Player Press": 1, "Supported Press": 2, "Organized Press": 3}
            
            current_event = {
                "start_window": smoothed_windows[0]["window"],
                "end_window": smoothed_windows[0]["window"],
                "possessing_team": smoothed_windows[0]["possessing_team"],
                "defending_team": smoothed_windows[0]["defending_team"],
                "start_zone": smoothed_windows[0]["pressing_zone"],
                "end_zone": smoothed_windows[0]["pressing_zone"],
                "max_pressing_style": smoothed_windows[0]["pressing_style"],
                "duration_windows": 1,
                "outcome": "Ongoing"
            }
            
            for i in range(1, len(smoothed_windows)):
                sw = smoothed_windows[i]
                
                # Check if possession flipped
                if sw["possessing_team"] != current_event["possessing_team"]:
                    current_event["outcome"] = "Success (Ball Recovered)"
                    pressing_events.append(current_event)
                    
                    # Start new sequence
                    current_event = {
                        "start_window": sw["window"],
                        "end_window": sw["window"],
                        "possessing_team": sw["possessing_team"],
                        "defending_team": sw["defending_team"],
                        "start_zone": sw["pressing_zone"],
                        "end_zone": sw["pressing_zone"],
                        "max_pressing_style": sw["pressing_style"],
                        "duration_windows": 1,
                        "outcome": "Ongoing"
                    }
                else:
                    # Possession retained. Update current sequence
                    current_event["end_window"] = sw["window"]
                    if sw["pressing_zone"]:
                        current_event["end_zone"] = sw["pressing_zone"]
                    current_event["duration_windows"] += 1
                    
                    if style_hierarchy.get(sw["pressing_style"], 0) > style_hierarchy.get(current_event["max_pressing_style"], 0):
                        current_event["max_pressing_style"] = sw["pressing_style"]
                        
                    # Check if ball escaped to a different zone
                    if current_event["start_zone"] and sw["pressing_zone"] and current_event["start_zone"] != sw["pressing_zone"]:
                        current_event["outcome"] = f"Failed (Ball Escaped to {sw['pressing_zone']})"
                        pressing_events.append(current_event)
                        
                        # Start new sequence in new zone
                        current_event = {
                            "start_window": sw["window"],
                            "end_window": sw["window"],
                            "possessing_team": sw["possessing_team"],
                            "defending_team": sw["defending_team"],
                            "start_zone": sw["pressing_zone"],
                            "end_zone": sw["pressing_zone"],
                            "max_pressing_style": sw["pressing_style"],
                            "duration_windows": 1,
                            "outcome": "Ongoing"
                        }
            
            pressing_events.append(current_event)

        # ── Aggregate Effectiveness Statistics ──
        stats = {
            "Team 1": {"total_presses": 0, "successful_presses": 0, "failed_presses": 0},
            "Team 2": {"total_presses": 0, "successful_presses": 0, "failed_presses": 0}
        }
        
        for ev in pressing_events:
            dt = ev["defending_team"]
            if ev["max_pressing_style"] != "No Press":
                stats[dt]["total_presses"] += 1
                if "Success" in ev["outcome"]:
                    stats[dt]["successful_presses"] += 1
                elif "Failed" in ev["outcome"]:
                    stats[dt]["failed_presses"] += 1

        summary = {}
        for team in ["Team 1", "Team 2"]:
            ts = stats[team]
            if ts["total_presses"] > 0:
                summary[team] = {
                    "total_pressing_sequences": ts["total_presses"],
                    "success_rate": round((ts["successful_presses"] / ts["total_presses"]) * 100, 2),
                    "failure_rate_due_to_escape": round((ts["failed_presses"] / ts["total_presses"]) * 100, 2)
                }
            else:
                summary[team] = "No active pressing sequences detected."

        return {
            "effectiveness_summary": summary,
            "pressing_events_timeline": pressing_events
        }

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_file = os.path.join(current_dir, "../output/tracking_tracking.db")
    analyzer = PressingAnalyzer(db_file)
    print("Analyzing pressing sequences and effectiveness...")
    match_data = analyzer.analyze_match(window_size=3)
    
    import json
    print("\n--- Effectiveness Summary ---")
    print(json.dumps(match_data["effectiveness_summary"], indent=2))
    
    print("\n--- Detailed Pressing Timeline ---")
    for event in match_data["pressing_events_timeline"]:
        print(json.dumps(event, indent=2))
