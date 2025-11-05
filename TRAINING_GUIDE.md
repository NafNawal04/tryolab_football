# Training Guide for ball.pt Model

This guide explains how to train a custom YOLOv5 model for soccer ball detection, based on the methodology described in the [Tryolabs blog post](https://tryolabs.com/blog/2022/10/17/measuring-soccer-ball-possession-ai-video-analytics).

## Overview

The `ball.pt` model was trained by:
1. Starting with YOLOv5 pretrained on COCO dataset
2. Creating a custom dataset from live soccer footage
3. Annotating the dataset using LabelImg
4. Fine-tuning YOLOv5 following the official YOLOv5 repository instructions

## Prerequisites

1. **Python Environment**: Python 3.8+
2. **YOLOv5**: Clone the official repository
3. **LabelImg**: For dataset annotation
4. **Soccer Video Footage**: Videos with similar camera views

## Step 1: Install YOLOv5

```bash
# Clone YOLOv5 repository
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

## Step 2: Prepare Your Dataset

### 2.1 Collect Soccer Videos
- Gather soccer match videos with similar camera angles
- Extract frames from videos (or use video directly)
- Ensure consistent lighting and camera perspective

### 2.2 Install LabelImg

```bash
# Install LabelImg for annotation
pip install labelimg
# Or using conda
conda install -c conda-forge labelimg
```

### 2.3 Annotate Your Dataset

1. Open LabelImg:
   ```bash
   labelimg
   ```

2. Set format to YOLO:
   - Go to "View" → Check "Auto Save"
   - Change format to YOLO (not PascalVOC)

3. Create directory structure:
   ```
   dataset/
   ├── images/
   │   ├── train/
   │   ├── val/
   │   └── test/
   └── labels/
       ├── train/
       ├── val/
       └── test/
   ```

4. Annotate images:
   - Open images from `images/train/`
   - Draw bounding boxes around soccer balls
   - Save labels (they'll be saved in `labels/train/` with same filename)
   - Repeat for validation and test sets

5. Create `dataset.yaml`:
   ```yaml
   # dataset.yaml
   path: /path/to/dataset  # dataset root dir
   train: images/train  # train images (relative to 'path')
   val: images/val  # val images (relative to 'path')
   test: images/test  # test images (optional)

   # Classes
   nc: 1  # number of classes
   names: ['ball']  # class names
   ```

## Step 3: Train the Model

### 3.1 Basic Training Command

```bash
cd yolov5
python train.py \
    --img 640 \
    --batch 16 \
    --epochs 100 \
    --data /path/to/dataset.yaml \
    --weights yolov5s.pt \
    --cache
```

### 3.2 Training Parameters Explained

- `--img 640`: Image size (640x640 pixels)
- `--batch 16`: Batch size (adjust based on GPU memory)
- `--epochs 100`: Number of training epochs
- `--data`: Path to your dataset.yaml file
- `--weights yolov5s.pt`: Starting weights (yolov5s = small, yolov5m = medium, yolov5l = large, yolov5x = extra large)
- `--cache`: Cache images for faster training

### 3.3 Recommended Training Setup

For ball detection (small objects), consider:
- **Model size**: Start with `yolov5s.pt` or `yolov5m.pt` (smaller models train faster)
- **Image size**: 640 or 1280 (larger helps detect small objects)
- **Batch size**: Adjust based on GPU memory (8-32)
- **Epochs**: 100-300 (monitor for overfitting)

### 3.4 Advanced Training Options

```bash
python train.py \
    --img 1280 \
    --batch 8 \
    --epochs 200 \
    --data /path/to/dataset.yaml \
    --weights yolov5s.pt \
    --hyp hyp.scratch.yaml \
    --multi-scale \
    --augment \
    --cache \
    --workers 8 \
    --project runs/train \
    --name ball_detection
```

Options:
- `--multi-scale`: Train with multiple image sizes
- `--augment`: Enable data augmentation
- `--workers 8`: Number of data loading workers
- `--project`: Project directory
- `--name`: Experiment name

## Step 4: Monitor Training

Training will create:
- `runs/train/ball_detection/weights/best.pt`: Best model weights
- `runs/train/ball_detection/weights/last.pt`: Last checkpoint
- Training metrics and plots in the same directory

Monitor:
- Loss curves (should decrease)
- mAP (mean Average Precision) - should increase
- Precision and Recall metrics

## Step 5: Validate Your Model

```bash
python val.py \
    --weights runs/train/ball_detection/weights/best.pt \
    --data /path/to/dataset.yaml \
    --img 640
```

## Step 6: Test Your Model

```bash
python detect.py \
    --weights runs/train/ball_detection/weights/best.pt \
    --source /path/to/test/video.mp4 \
    --conf 0.3 \
    --save-txt \
    --save-conf
```

## Step 7: Use Your Trained Model

Copy your trained model to the project:
```bash
cp runs/train/ball_detection/weights/best.pt /path/to/soccer-video-analytics/models/ball.pt
```

Then use it with the existing code:
```bash
python run.py --possession --model models/ball.pt --video videos/your_video.mp4
```

## Tips for Better Results

1. **Dataset Quality**:
   - Annotate at least 500-1000 images
   - Include diverse scenarios (different lighting, ball sizes, angles)
   - Ensure consistent annotations

2. **Data Augmentation**:
   - Enable augmentation in YOLOv5 (already included)
   - Helps with generalization

3. **Hyperparameter Tuning**:
   - Adjust learning rate if loss doesn't decrease
   - Try different model sizes (s, m, l, x)
   - Experiment with image sizes

4. **Overfitting Prevention**:
   - Use validation split (typically 80/20 or 70/30)
   - Monitor validation loss
   - Early stopping if validation loss stops improving

5. **Small Object Detection**:
   - Use larger image sizes (1280 instead of 640)
   - Consider using YOLOv5m or YOLOv5l (better for small objects)
   - Ensure good annotation quality

## Troubleshooting

**Issue**: Model not detecting balls
- **Solution**: Check annotations, increase image size, train longer

**Issue**: Too many false positives
- **Solution**: Lower confidence threshold, add more negative examples, train longer

**Issue**: Training too slow
- **Solution**: Reduce batch size, use smaller model (yolov5s), reduce image size

**Issue**: Out of memory
- **Solution**: Reduce batch size, use smaller image size, use smaller model

## Additional Resources

- [YOLOv5 Documentation](https://docs.ultralytics.com/)
- [YOLOv5 GitHub](https://github.com/ultralytics/yolov5)
- [LabelImg GitHub](https://github.com/HumanSignal/labelImg)
- [Tryolabs Blog Post](https://tryolabs.com/blog/2022/10/17/measuring-soccer-ball-possession-ai-video-analytics)

## Notes

⚠️ **Important**: As mentioned in the blog post, the original `ball.pt` is a toy model that overfits to specific videos. For production use, you'll need:
- A larger, more diverse dataset
- Careful validation and testing
- Potentially multiple training iterations
- Consider using transfer learning from other object detection models

