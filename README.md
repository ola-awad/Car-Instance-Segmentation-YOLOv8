# 🚗 Car Instance Segmentation using YOLOv8

An end-to-end Computer Vision project focused on precise car instance segmentation using **YOLOv8s-seg**. This project tackles common computer vision challenges such as **Mask Overlap** and boundaries bleeding into low-contrast road surfaces.

---

## 📌 Project Overview
The objective of this project is to accurately detect and segment individual cars in varying road and light conditions. 

Initially, lighter models like `yolov8n-seg` struggled with boundary precision and mask overlap (segmentation mask spilling onto asphalt due to low contrast). Upgrading to `YOLOv8s-seg` provided superior feature extraction, resulting in crisp and well-defined segmentation boundaries.

---

## 📊 Performance & Metrics

After training for **100 epochs**, the model achieved exceptional results across all validation metrics:

| Metric | Precision (P) | Recall (R) | mAP@50 | mAP@50-95 |
| :--- | :---: | :---: | :---: | :---: |
| **Bounding Box** | **99.8%** | **92.2%** | **95.8%** | **93.8%** |
| **Mask Segmentation** | **99.8%** | **92.2%** | **95.8%** | **91.0%** |

---

## 📸 Inference Results

| Input & Output Segmentation Sample |
| :---: |
| ![Segmentation Result](assets/sample_result1.jpg) |

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Car-Instance-Segmentation-YOLOv8.git](https://github.com/YOUR_USERNAME/Car-Instance-Segmentation-YOLOv8.git)
   cd Car-Instance-Segmentation-YOLOv8

2. **Install dependencies:**
   ```bash
   pip install ultralytics opencv-python
   ```

3. **Run Inference with Trained Weights:**
   ```python
   from ultralytics import YOLO

   # Load trained weights
   model = YOLO('weights/best.pt')

   # Predict on new images
   results = model.predict(source='path/to/test/images', conf=0.3, save=True)
   ```
