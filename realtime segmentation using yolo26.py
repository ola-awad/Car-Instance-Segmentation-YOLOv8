import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # Fix for Intel MKL duplicate library issue

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# ============================
# Load YOLOv8 Segmentation Model
# ============================
model = YOLO("best.pt")   # You can replace with your custom best.pt

# ============================
# Open input video
# ============================
cap = cv2.VideoCapture("traffictrim.mp4")

# Get video width, height, and fps
w, h, fps = (int(cap.get(x)) for x in (
    cv2.CAP_PROP_FRAME_WIDTH,
    cv2.CAP_PROP_FRAME_HEIGHT,
    cv2.CAP_PROP_FPS
))

# ============================
# Create output video writer
# ============================
out = cv2.VideoWriter(
    'instance-segmentation-object-tracking.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    fps,
    (w, h)
)

# ============================
# Process each frame
# ============================
while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or processing completed.")
        break

    # Run model tracking + segmentation
    results = model.track(
        im0,
        iou=0.5,
        show=False,
        persist=True,
        tracker="bytetrack.yaml"
    )

    # If results exist, let YOLO handle the full annotation automatically
    if results[0].boxes.id is not None and results[0].masks is not None:
        # This replaces the entire annotator loop and cv2.polylines logic
        im0 = results[0].plot() 

    # Write frame to output video
    out.write(im0)

    # Show frame on screen
    cv2.imshow("instance-segmentation-object-tracking", im0)

    # Quit if user presses Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break