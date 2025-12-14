"""
Test script to verify keypoint detection and tactical view conversion.
This script processes a few frames from the video and outputs debug information.
"""
import cv2
import numpy as np
from court_keypoint_detector import CourtKeypointDetector
from tactical_view import TacticalViewProjector

def test_keypoint_detection(video_path, num_frames=10):
    """Test keypoint detection on sample frames."""
    print(f"Testing keypoint detection on {video_path}")
    print("=" * 60)
    
    # Initialize detector and projector
    court_detector = CourtKeypointDetector(model_path="models/keypoint_detector.pt")
    tactical_projector = TacticalViewProjector(width=400, height=240)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    frame_count = 0
    successful_detections = 0
    successful_homographies = 0
    
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        print(f"\nFrame {frame_count}:")
        print("-" * 40)
        
        # Detect keypoints
        court_keypoints = court_detector.get_court_keypoints([frame])
        
        if court_keypoints and len(court_keypoints) > 0:
            kp = court_keypoints[0]
            
            # Check if keypoints exist
            if hasattr(kp, 'xy') and kp.xy is not None:
                kp_list = kp.xy.tolist()
                if kp_list and len(kp_list[0]) > 0:
                    frame_keypoints = kp_list[0]
                    valid_kps = [(kp[0], kp[1]) for kp in frame_keypoints if kp[0] > 0 and kp[1] > 0]
                    
                    print(f"  ✓ Detected {len(valid_kps)} valid keypoints out of {len(frame_keypoints)} total")
                    successful_detections += 1
                    
                    # Try to update homography
                    success = tactical_projector.update_homography_from_keypoints(court_keypoints[0])
                    if success:
                        print(f"  ✓ Homography updated successfully")
                        successful_homographies += 1
                        
                        # Show bounding box of detected keypoints
                        if valid_kps:
                            min_x = min(kp[0] for kp in valid_kps)
                            max_x = max(kp[0] for kp in valid_kps)
                            min_y = min(kp[1] for kp in valid_kps)
                            max_y = max(kp[1] for kp in valid_kps)
                            print(f"  Field bounds: x=[{min_x:.1f}, {max_x:.1f}], y=[{min_y:.1f}, {max_y:.1f}]")
                    else:
                        print(f"  ✗ Homography update failed")
                else:
                    print(f"  ✗ No valid keypoints detected")
            else:
                print(f"  ✗ Keypoints object has no xy attribute")
        else:
            print(f"  ✗ No keypoints returned")
    
    cap.release()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total frames processed: {frame_count}")
    print(f"  Successful keypoint detections: {successful_detections} ({successful_detections/frame_count*100:.1f}%)")
    print(f"  Successful homography updates: {successful_homographies} ({successful_homographies/frame_count*100:.1f}%)")
    print("=" * 60)

if __name__ == "__main__":
    import sys
    
    video_path = "videos/soccer_possession1.mp4"
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    
    test_keypoint_detection(video_path, num_frames=30)
