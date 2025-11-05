#!/usr/bin/env python3
"""
Helper script to prepare dataset for YOLOv5 training.
Extracts frames from videos and creates directory structure for annotation.
"""

import argparse
import os
import cv2
from pathlib import Path


def extract_frames(video_path, output_dir, interval=30):
    """
    Extract frames from video at specified intervals.
    
    Parameters
    ----------
    video_path : str
        Path to input video
    output_dir : str
        Directory to save frames
    interval : int
        Extract every Nth frame (default: 30, meaning 1 frame per second at 30fps)
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {video_path.name}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Extracting every {interval} frames...")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame at specified interval
        if frame_count % interval == 0:
            frame_filename = f"{video_path.stem}_frame_{frame_count:06d}.jpg"
            frame_path = output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames to {output_dir}")
    return saved_count


def create_directory_structure(base_dir):
    """
    Create YOLOv5 dataset directory structure.
    
    Parameters
    ----------
    base_dir : str
        Base directory for dataset
    """
    base_dir = Path(base_dir)
    
    directories = [
        base_dir / "images" / "train",
        base_dir / "images" / "val",
        base_dir / "images" / "test",
        base_dir / "labels" / "train",
        base_dir / "labels" / "val",
        base_dir / "labels" / "test",
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory structure in {base_dir}")
    return base_dir


def create_dataset_yaml(dataset_dir, output_path=None):
    """
    Create dataset.yaml file for YOLOv5.
    
    Parameters
    ----------
    dataset_dir : str
        Path to dataset directory
    output_path : str, optional
        Path to save yaml file (default: dataset_dir/dataset.yaml)
    """
    dataset_dir = Path(dataset_dir).resolve()
    
    if output_path is None:
        output_path = dataset_dir / "dataset.yaml"
    else:
        output_path = Path(output_path)
    
    yaml_content = f"""# Dataset configuration for YOLOv5
# Path to dataset (absolute or relative to this file)
path: {dataset_dir}

# Train/val/test splits (relative to 'path')
train: images/train
val: images/val
test: images/test  # optional

# Number of classes
nc: 1

# Class names
names:
  0: ball
"""
    
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created dataset.yaml at {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for YOLOv5 ball detection training"
    )
    parser.add_argument(
        "--create-structure",
        type=str,
        help="Create directory structure at specified path"
    )
    parser.add_argument(
        "--extract-frames",
        type=str,
        help="Extract frames from video file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/images/train",
        help="Output directory for frames (default: dataset/images/train)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Extract every Nth frame (default: 30)"
    )
    parser.add_argument(
        "--create-yaml",
        type=str,
        help="Create dataset.yaml file for specified dataset directory"
    )
    
    args = parser.parse_args()
    
    if args.create_structure:
        create_directory_structure(args.create_structure)
        if args.create_yaml is None:
            create_dataset_yaml(args.create_structure)
    
    if args.extract_frames:
        extract_frames(args.extract_frames, args.output_dir, args.interval)
    
    if args.create_yaml:
        create_dataset_yaml(args.create_yaml)
    
    if not any([args.create_structure, args.extract_frames, args.create_yaml]):
        parser.print_help()


if __name__ == "__main__":
    main()

