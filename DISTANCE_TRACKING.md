# Player Distance Tracking Implementation

## Overview
This implementation adds cumulative distance tracking for players in the soccer video analysis pipeline. Distances are tracked using stabilized coordinates (from motion compensation) and can optionally be converted to real-world meters.

## What Was Implemented

### 1. Player Class Enhancements (`soccer/player.py`)
- Added `center` property: Returns player's center position in stabilized coordinates (pixels)
- Added `center_abs` property: Returns player's center position in absolute coordinates (pixels)
- Added `player_id` property: Returns the player's tracking ID

### 2. PlayerDistanceTracker Class (`soccer/match.py`)
A new class that tracks cumulative distances for all players:
- Tracks players by their tracking ID across frames
- Uses stabilized coordinates (`detection.points`) for consistent distance calculations
- Stores last position and cumulative distance for each player
- Supports optional conversion from pixels to meters

### 3. Match Class Enhancements (`soccer/match.py`)
- Added `pixels_to_meters` parameter to `Match.__init__()` for optional calibration
- Added `distance_tracker` attribute: `PlayerDistanceTracker` instance
- Updated `Match.update()` to automatically track distances for all players each frame
- Added methods:
  - `get_player_distance(player, in_meters=False)`: Get cumulative distance for a player
  - `get_team_total_distance(players, team, in_meters=False)`: Get total distance for a team
  - `get_all_distances(in_meters=False)`: Get distances for all tracked players
  - `get_distance_statistics(in_meters=False)`: Get statistics (total, mean, min, max, median, count)
  - `reset_distance_tracking()`: Reset all distance data

## How It Works

1. **Coordinate System**: Uses stabilized coordinates (`detection.points`) which are motion-compensated to provide consistent frame-to-frame distances
2. **Distance Calculation**: Euclidean distance between consecutive frame positions
3. **Tracking**: Players are tracked by their ID from the detection tracker, so distances persist across frames even if players temporarily disappear
4. **Accumulation**: Distances are accumulated frame-by-frame for each player

## Usage

### Basic Usage (Pixels Only)
```python
# Create match (distances in pixels)
match = Match(home=home_team, away=away_team, fps=30)

# After processing frames, get distances
for player in players:
    distance_pixels = match.get_player_distance(player)
    print(f"Player {player.player_id}: {distance_pixels:.2f} pixels")

# Get statistics
stats = match.get_distance_statistics()
print(f"Total distance: {stats['total']:.2f} pixels")
print(f"Mean distance: {stats['mean']:.2f} pixels")
```

### Usage with Meters Conversion
```python
# Calculate pixels_to_meters conversion factor
# Example: If you know that 100 pixels = 1 meter, use 0.01
# Or if you calibrate using field dimensions:
# pitch_length_meters = 105  # Standard soccer field length
# pitch_length_pixels = 1050  # Measured from video
# pixels_to_meters = pitch_length_meters / pitch_length_pixels

match = Match(
    home=home_team, 
    away=away_team, 
    fps=30,
    pixels_to_meters=0.01  # 100 pixels = 1 meter
)

# Get distances in meters
distance_meters = match.get_player_distance(player, in_meters=True)
stats = match.get_distance_statistics(in_meters=True)
```

### Getting Team Distances
```python
# Get total distance for a team (requires current players list)
home_distance = match.get_team_total_distance(players, match.home, in_meters=True)
away_distance = match.get_team_total_distance(players, match.away, in_meters=True)
```

## What You Need to Bring

### 1. Calibration Data (Optional - for meters conversion)
To convert pixel distances to real-world meters, you need:

**Option A: Manual Calibration**
- Measure a known distance on the field in the video (e.g., penalty box width = 16.5m)
- Measure the same distance in pixels in the stabilized coordinate system
- Calculate: `pixels_to_meters = known_meters / measured_pixels`

**Option B: Field Dimensions**
- Standard soccer field dimensions:
  - Length: 100-110 meters
  - Width: 64-75 meters
- Measure field dimensions in pixels from the video
- Calculate conversion factor for length and width (may differ slightly)

**Option C: Camera Calibration**
- Use camera calibration techniques (chessboard calibration, etc.)
- Derive homography matrix from camera to field plane
- Apply homography to convert pixel coordinates to field coordinates (meters)

### 2. Coordinate System Considerations
- **Stabilized coordinates**: Currently uses `detection.points` which are motion-compensated but still in pixel space
- **Field coordinates**: For true real-world distances, you'd need to:
  1. Use `coord_transformations.abs_to_rel()` to get stabilized coordinates
  2. Apply a pitch homography to convert to normalized field coordinates
  3. Scale to meters using known field dimensions

### 3. Integration with Existing Pipeline
The distance tracking is automatically integrated into `Match.update()`, so no changes are needed to `run.py` unless you want to:
- Add command-line argument for `pixels_to_meters`
- Log/display distances during processing
- Export distance statistics to a file

## Example: Adding Calibration to run.py

```python
# Add argument for pixels_to_meters
parser.add_argument(
    "--pixels-to-meters",
    type=float,
    default=None,
    help="Conversion factor from pixels to meters (e.g., 0.01 for 100px=1m)"
)

# Update build_match_setup
def build_match_setup(
    match_key: str,
    fps: float,
    pixels_to_meters: Optional[float] = None,
):
    # ... existing code ...
    match = Match(
        home=home,
        away=away,
        fps=fps,
        pixels_to_meters=pixels_to_meters,
    )
    # ... rest of code ...
```

## Notes

1. **Accuracy**: Distance accuracy depends on:
   - Quality of player tracking (ID consistency)
   - Motion compensation accuracy
   - Frame rate (higher FPS = more accurate)
   - Calibration accuracy (if using meters)

2. **Coordinate System**: Currently uses stabilized pixel coordinates. For true real-world distances, additional calibration (homography) is needed.

3. **Player Tracking**: Players must have consistent IDs from the tracker. If IDs change frequently, distances may be inaccurate.

4. **Performance**: Distance tracking adds minimal overhead (simple distance calculations per frame).

## Future Enhancements

1. **Pitch Homography**: Integrate homography transformation to convert directly to field coordinates
2. **Speed/Acceleration**: Calculate player speed and acceleration from distances
3. **Heat Maps**: Generate heat maps from player positions
4. **Distance Zones**: Track distances in different zones (defensive, midfield, attacking)
5. **Export**: Export distance data to CSV/JSON for analysis

