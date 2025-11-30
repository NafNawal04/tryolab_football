"""
Movement Analysis Module

Tracks player paths, off-ball runs, and zone transitions (defensive, midfield, attacking)
using tactical view coordinates from homography projection.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from soccer.player import Player
from soccer.team import Team


class Zone(Enum):
    """Pitch zones based on tactical view x-coordinate."""
    DEFENSIVE = "defensive"
    MIDFIELD = "midfield"
    ATTACKING = "attacking"


@dataclass
class PositionRecord:
    """Record of a player's position at a specific frame."""
    frame_number: int
    tactical_x: float
    tactical_y: float
    zone: Zone
    has_ball: bool
    speed: float  # pixels per frame


@dataclass
class OffBallRun:
    """Represents an off-ball run segment."""
    player_id: int
    start_frame: int
    end_frame: int
    start_zone: Zone
    end_zone: Zone
    distance: float  # in tactical view pixels
    duration_frames: int
    avg_speed: float


@dataclass
class ZoneTransition:
    """Represents a transition between zones."""
    player_id: int
    frame_number: int
    from_zone: Zone
    to_zone: Zone
    direction: str  # "forward" or "backward"


@dataclass
class PlayerMovementStats:
    """Aggregated movement statistics for a player."""
    player_id: int
    team: Optional[Team]
    
    # Zone statistics
    time_in_defensive: int  # frames
    time_in_midfield: int
    time_in_attacking: int
    
    # Zone transitions
    defensive_to_midfield: int
    midfield_to_attacking: int
    attacking_to_midfield: int
    midfield_to_defensive: int
    
    # Off-ball runs
    off_ball_runs: List[OffBallRun]
    total_off_ball_distance: float
    
    # Path statistics
    total_path_length: float
    max_speed: float
    avg_speed: float


class MovementAnalyzer:
    """
    Analyzes player movement patterns, paths, off-ball runs, and zone transitions.
    Uses tactical view coordinates for consistent zone definitions.
    """
    
    def __init__(
        self,
        tactical_width: int = 400,
        tactical_height: int = 200,
        min_run_frames: int = 10,
        min_run_distance: float = 20.0,
    ):
        """
        Initialize movement analyzer.
        
        Parameters
        ----------
        tactical_width : int
            Width of tactical view in pixels (default: 400)
        tactical_height : int
            Height of tactical view in pixels (default: 200)
        min_run_frames : int
            Minimum frames for an off-ball run to be counted (default: 10)
        min_run_distance : float
            Minimum distance in tactical pixels for an off-ball run (default: 20.0)
        """
        self.tactical_width = tactical_width
        self.tactical_height = tactical_height
        
        # Zone boundaries (based on x-coordinate in tactical view)
        # Defensive: 0 to 1/3, Midfield: 1/3 to 2/3, Attacking: 2/3 to 1.0
        self.defensive_boundary = tactical_width / 3
        self.midfield_boundary = 2 * tactical_width / 3
        
        self.min_run_frames = min_run_frames
        self.min_run_distance = min_run_distance
        
        # Store player paths: player_id -> List[PositionRecord]
        self.player_paths: Dict[int, List[PositionRecord]] = {}
        
        # Store current frame number
        self.current_frame = 0
        
        # Store team information for players
        self.player_teams: Dict[int, Optional[Team]] = {}
    
    def _get_zone(self, tactical_x: float) -> Zone:
        """Determine zone based on tactical x-coordinate."""
        if tactical_x < self.defensive_boundary:
            return Zone.DEFENSIVE
        elif tactical_x < self.midfield_boundary:
            return Zone.MIDFIELD
        else:
            return Zone.ATTACKING
    
    def _calculate_speed(
        self,
        pos1: Optional[PositionRecord],
        pos2: PositionRecord,
    ) -> float:
        """Calculate speed (distance per frame) between two positions."""
        if pos1 is None:
            return 0.0
        
        dx = pos2.tactical_x - pos1.tactical_x
        dy = pos2.tactical_y - pos1.tactical_y
        distance = np.sqrt(dx * dx + dy * dy)
        
        frames_diff = pos2.frame_number - pos1.frame_number
        if frames_diff == 0:
            return 0.0
        
        return distance / frames_diff
    
    def update(
        self,
        players: List[Player],
        tactical_positions: Dict[int, Tuple[float, float]],
        closest_player: Optional[Player],
    ):
        """
        Update movement analysis with current frame data.
        
        Parameters
        ----------
        players : List[Player]
            Current list of players
        tactical_positions : Dict[int, Tuple[float, float]]
            Dictionary mapping player_id to (tactical_x, tactical_y) coordinates
        closest_player : Optional[Player]
            Player closest to ball (has possession), or None
        """
        self.current_frame += 1
        
        closest_player_id = closest_player.player_id if closest_player else None
        
        for player in players:
            if player.player_id is None:
                continue
            
            player_id = player.player_id
            
            # Store team information
            if player_id not in self.player_teams:
                self.player_teams[player_id] = player.team
            
            # Get tactical position
            if player_id not in tactical_positions:
                continue
            
            tactical_x, tactical_y = tactical_positions[player_id]
            
            # Determine zone
            zone = self._get_zone(tactical_x)
            
            # Check if player has ball
            has_ball = (player_id == closest_player_id)
            
            # Get previous position for speed calculation
            previous_pos = None
            if player_id in self.player_paths and len(self.player_paths[player_id]) > 0:
                previous_pos = self.player_paths[player_id][-1]
            
            # Calculate speed
            current_pos = PositionRecord(
                frame_number=self.current_frame,
                tactical_x=tactical_x,
                tactical_y=tactical_y,
                zone=zone,
                has_ball=has_ball,
                speed=0.0,  # Will calculate below
            )
            
            speed = self._calculate_speed(previous_pos, current_pos)
            current_pos.speed = speed
            
            # Add to path
            if player_id not in self.player_paths:
                self.player_paths[player_id] = []
            
            self.player_paths[player_id].append(current_pos)
    
    def _detect_off_ball_runs(self, player_id: int) -> List[OffBallRun]:
        """Detect off-ball runs for a specific player."""
        if player_id not in self.player_paths:
            return []
        
        path = self.player_paths[player_id]
        if len(path) < 2:
            return []
        
        runs: List[OffBallRun] = []
        current_run_start: Optional[PositionRecord] = None
        current_run_frames = 0
        
        for i, pos in enumerate(path):
            if not pos.has_ball:
                # Continue or start a run
                if current_run_start is None:
                    current_run_start = pos
                    current_run_frames = 1
                else:
                    current_run_frames += 1
            else:
                # End current run if exists
                if current_run_start is not None and current_run_frames >= self.min_run_frames:
                    # Calculate run distance
                    start_pos = current_run_start
                    end_pos = path[i - 1] if i > 0 else pos
                    
                    dx = end_pos.tactical_x - start_pos.tactical_x
                    dy = end_pos.tactical_y - start_pos.tactical_y
                    distance = np.sqrt(dx * dx + dy * dy)
                    
                    if distance >= self.min_run_distance:
                        avg_speed = distance / current_run_frames if current_run_frames > 0 else 0.0
                        
                        runs.append(OffBallRun(
                            player_id=player_id,
                            start_frame=start_pos.frame_number,
                            end_frame=end_pos.frame_number,
                            start_zone=start_pos.zone,
                            end_zone=end_pos.zone,
                            distance=distance,
                            duration_frames=current_run_frames,
                            avg_speed=avg_speed,
                        ))
                
                current_run_start = None
                current_run_frames = 0
        
        # Handle run that extends to end of path
        if current_run_start is not None and current_run_frames >= self.min_run_frames:
            start_pos = current_run_start
            end_pos = path[-1]
            
            dx = end_pos.tactical_x - start_pos.tactical_x
            dy = end_pos.tactical_y - start_pos.tactical_y
            distance = np.sqrt(dx * dx + dy * dy)
            
            if distance >= self.min_run_distance:
                avg_speed = distance / current_run_frames if current_run_frames > 0 else 0.0
                
                runs.append(OffBallRun(
                    player_id=player_id,
                    start_frame=start_pos.frame_number,
                    end_frame=end_pos.frame_number,
                    start_zone=start_pos.zone,
                    end_zone=end_pos.zone,
                    distance=distance,
                    duration_frames=current_run_frames,
                    avg_speed=avg_speed,
                ))
        
        return runs
    
    def _detect_zone_transitions(self, player_id: int) -> List[ZoneTransition]:
        """Detect zone transitions for a specific player."""
        if player_id not in self.player_paths:
            return []
        
        path = self.player_paths[player_id]
        if len(path) < 2:
            return []
        
        transitions: List[ZoneTransition] = []
        previous_zone = path[0].zone
        
        for pos in path[1:]:
            current_zone = pos.zone
            
            if current_zone != previous_zone:
                # Determine direction
                zone_order = [Zone.DEFENSIVE, Zone.MIDFIELD, Zone.ATTACKING]
                from_idx = zone_order.index(previous_zone)
                to_idx = zone_order.index(current_zone)
                
                direction = "forward" if to_idx > from_idx else "backward"
                
                transitions.append(ZoneTransition(
                    player_id=player_id,
                    frame_number=pos.frame_number,
                    from_zone=previous_zone,
                    to_zone=current_zone,
                    direction=direction,
                ))
                
                previous_zone = current_zone
        
        return transitions
    
    def get_player_stats(self, player_id: int) -> Optional[PlayerMovementStats]:
        """Get aggregated movement statistics for a specific player."""
        if player_id not in self.player_paths:
            return None
        
        path = self.player_paths[player_id]
        if len(path) == 0:
            return None
        
        # Count time in each zone
        time_in_defensive = sum(1 for p in path if p.zone == Zone.DEFENSIVE)
        time_in_midfield = sum(1 for p in path if p.zone == Zone.MIDFIELD)
        time_in_attacking = sum(1 for p in path if p.zone == Zone.ATTACKING)
        
        # Count zone transitions
        transitions = self._detect_zone_transitions(player_id)
        
        defensive_to_midfield = sum(
            1 for t in transitions
            if t.from_zone == Zone.DEFENSIVE and t.to_zone == Zone.MIDFIELD
        )
        midfield_to_attacking = sum(
            1 for t in transitions
            if t.from_zone == Zone.MIDFIELD and t.to_zone == Zone.ATTACKING
        )
        attacking_to_midfield = sum(
            1 for t in transitions
            if t.from_zone == Zone.ATTACKING and t.to_zone == Zone.MIDFIELD
        )
        midfield_to_defensive = sum(
            1 for t in transitions
            if t.from_zone == Zone.MIDFIELD and t.to_zone == Zone.DEFENSIVE
        )
        
        # Detect off-ball runs
        off_ball_runs = self._detect_off_ball_runs(player_id)
        total_off_ball_distance = sum(run.distance for run in off_ball_runs)
        
        # Calculate path statistics
        total_path_length = 0.0
        speeds = []
        
        for i in range(1, len(path)):
            prev_pos = path[i - 1]
            curr_pos = path[i]
            
            dx = curr_pos.tactical_x - prev_pos.tactical_x
            dy = curr_pos.tactical_y - prev_pos.tactical_y
            segment_length = np.sqrt(dx * dx + dy * dy)
            total_path_length += segment_length
            
            if curr_pos.speed > 0:
                speeds.append(curr_pos.speed)
        
        max_speed = max(speeds) if speeds else 0.0
        avg_speed = np.mean(speeds) if speeds else 0.0
        
        return PlayerMovementStats(
            player_id=player_id,
            team=self.player_teams.get(player_id),
            time_in_defensive=time_in_defensive,
            time_in_midfield=time_in_midfield,
            time_in_attacking=time_in_attacking,
            defensive_to_midfield=defensive_to_midfield,
            midfield_to_attacking=midfield_to_attacking,
            attacking_to_midfield=attacking_to_midfield,
            midfield_to_defensive=midfield_to_defensive,
            off_ball_runs=off_ball_runs,
            total_off_ball_distance=total_off_ball_distance,
            total_path_length=total_path_length,
            max_speed=max_speed,
            avg_speed=avg_speed,
        )
    
    def get_all_stats(self) -> Dict[int, PlayerMovementStats]:
        """Get movement statistics for all tracked players."""
        stats = {}
        for player_id in self.player_paths.keys():
            stat = self.get_player_stats(player_id)
            if stat is not None:
                stats[player_id] = stat
        return stats
    
    def get_team_stats(self, team: Team) -> List[PlayerMovementStats]:
        """Get movement statistics for all players on a specific team."""
        all_stats = self.get_all_stats()
        return [
            stat for stat in all_stats.values()
            if stat.team is not None and stat.team == team
        ]
    
    def print_summary(self, fps: float = 30.0):
        """Print a summary of movement analysis."""
        all_stats = self.get_all_stats()
        
        if not all_stats:
            print("No movement data available.")
            return
        
        print("\n" + "=" * 80)
        print("MOVEMENT ANALYSIS SUMMARY")
        print("=" * 80)
        
        # Group by team (use team name as key since Team objects aren't hashable)
        team_stats: Dict[str, List[PlayerMovementStats]] = {}
        team_objects: Dict[str, Optional[Team]] = {}
        
        for stat in all_stats.values():
            team = stat.team
            team_name = team.name if team else "Unknown"
            
            if team_name not in team_stats:
                team_stats[team_name] = []
                team_objects[team_name] = team
            
            team_stats[team_name].append(stat)
        
        for team_name, stats_list in team_stats.items():
            print(f"\n{team_name.upper()} ({len(stats_list)} players)")
            print("-" * 80)
            
            # Sort by total path length (most active first)
            stats_list.sort(key=lambda s: s.total_path_length, reverse=True)
            
            for stat in stats_list[:10]:  # Top 10 most active
                total_frames = stat.time_in_defensive + stat.time_in_midfield + stat.time_in_attacking
                if total_frames == 0:
                    continue
                
                def_pct = (stat.time_in_defensive / total_frames) * 100
                mid_pct = (stat.time_in_midfield / total_frames) * 100
                att_pct = (stat.time_in_attacking / total_frames) * 100
                
                print(f"\n  Player {stat.player_id}:")
                print(f"    Zone Distribution: Def {def_pct:.1f}% | Mid {mid_pct:.1f}% | Att {att_pct:.1f}%")
                print(f"    Zone Transitions: {stat.defensive_to_midfield + stat.midfield_to_attacking + stat.attacking_to_midfield + stat.midfield_to_defensive}")
                print(f"    Off-ball Runs: {len(stat.off_ball_runs)} (total distance: {stat.total_off_ball_distance:.1f} px)")
                print(f"    Total Path: {stat.total_path_length:.1f} px | Avg Speed: {stat.avg_speed:.2f} px/frame")
        
        print("\n" + "=" * 80)

