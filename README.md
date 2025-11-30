# Evaluating Synthetic Data Training, Model Architecture, and Quantization for CCTV Analytics on Raspberry Pi 5

This repository contains the complete codebase, experiments, datasets, scripts, and documentation for a Final Year Project investigating whether synthetic CCTV data can effectively train person-detection models comparable to those trained on real CCTV footage. The project also explores model architecture efficiency, INT8 quantization, and edge deployment on Raspberry Pi 5.

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Project Structure](#-project-structure)
- [Key Experiments](#-key-experiments)
- [Dataset Details](#-dataset-details)
- [Environment Setup](#-environment-setup)
- [Installation](#-installation)
- [Pipeline Workflow](#-pipeline-workflow)
- [Training Models](#-training-models)
- [Reproducing Experiments](#-reproducing-experiments)
- [Results Summary](#-results-summary)
- [Future Work](#-future-work)

---

## 🎯 Project Overview

### High-Level Summary

This project investigates three critical questions for edge AI deployment:

1. **Can synthetic data replace real CCTV footage for training?** (Experiment 1)
2. **How does INT8 quantization affect model performance?** (Experiment 2)
3. **What's the optimal architecture for edge deployment?** (Experiment 3)

All experiments use YOLO-based object detection, real CCTV clips, synthetic images generated with NanoBanana (Google Gemini), and the Ultralytics training pipeline.

### Research Questions

- **Experiment 1**: Does synthetic CCTV data perform as well as real CCTV data for person detection?
- **Experiment 2**: Do synthetic-trained and real-trained models degrade differently under INT8 quantization?
- **Experiment 3**: Is object detection or image classification better for real-time CCTV analytics on Raspberry Pi 5?

---

## 📁 Project Structure

```
FYP/
├── configs/                      # YOLO dataset configuration files
│   ├── mydata.yaml              # Original dataset config
│   ├── mydata_capped.yaml       # Capped real data config (train/val/test)
│   └── mydata_synthetic.yaml    # Synthetic data config
├── data/                        # Dataset split files
│   ├── train_capped.txt         # Training split (1,795 images)
│   ├── val_capped.txt           # Validation split (664 images)
│   ├── test_capped.txt          # Test split (660 images)
│   ├── train_fake.txt           # Synthetic training split
│   └── val_fake.txt             # Synthetic validation split
├── notebooks/                   # Jupyter notebooks for training
│   └── Experiment_1.ipynb       # Experiment 1 training code
├── runs/                        # Training outputs and results
│   ├── real_unfrozen_e50/       # Real data model training run
│   └── synthetic_unfrozen_e40/  # Synthetic data model training run
└── scripts/                     # Preprocessing and data pipeline scripts
    ├── dedupe_by_threshold.py           # Near-duplicate removal using SSCD/FAISS
    ├── assign_clusters.py               # SSCD scene clustering assignments
    ├── split.py                         # Train/val/test split creation with capping
    ├── generate_synthetic_backgrounds.py # Synthetic background generation
    ├── fake_background_synthetic_person_placement.py # Synthetic person placement
    └── batch_background_removal.py      # Background extraction from real images
```

---

## 🔬 Key Experiments

### Experiment 1: Real vs Synthetic CCTV Training

**Goal**: Determine whether synthetic data can replace or match real CCTV performance for person detection.

#### Process Overview

1. **Data Collection**

   - Collected ~22,000 real CCTV frames from multiple sources:
     - 1,637 YouTube CCTV videos
     - 11,722 home CCTV frames
     - 8,807 company CCTV frames
2. **Synthetic Data Generation**

   - Generated ~6,000 synthetic CCTV images using Image-to-Image GenAI:
     - 50 day backgrounds + 50 night backgrounds
     - ~6,000 generated persons with hybrid image+text prompts
     - Used NanoBanana (Google Gemini) for all synthetic generation
     - No alteration of backgrounds (critical for fair comparison)
     - Training set size matched to real dataset: 1,795 images for fair comparison
3. **Preprocessing Pipeline**

   - **Annotation**: Done first on all images using CVAT before any preprocessing steps
   - **Deduplication**: Used cosine similarity + SSCD embeddings to remove near-identical frames
   - **Scene Clustering**: Used SSCD (Scene Similarity Clustering Dataset) to group similar camera angles
   - **Data Capping**: Capped nighttime clusters to avoid data imbalance
   - **Train/Val/Test Split**: Ensured leakage-free splits per-cluster (no data leakage)
4. **Training Configuration**

   - Trained two YOLO11n models with identical hyperparameters:
     - **Real-only model**: Trained on real CCTV data
     - **Synthetic-only model**: Trained on synthetic CCTV data
   - Both models used:
     - `imgsz=832` (high resolution for small objects)
     - `batch=16`
     - `lr=2e-4`
     - `epochs=40`
     - Backbone unfrozen
     - RTX 4090 via Vast.ai
5. **Evaluation**

   - Evaluated both models on the same real CCTV test set only
   - Metrics: Precision, Recall, mAP@50, mAP@50-95

#### Results

- **Real-trained model performed best**
- **Synthetic-only model underperformed**
- **Clear domain gap exists** between synthetic and real CCTV scenes
- **Conclusion**: Synthetic data cannot fully replace real data at this stage, though it may be useful for augmentation

---

### Experiment 2: Quantization & Edge Efficiency (Coming Next)

**Goal**: Evaluate whether INT8 quantization affects synthetic-trained and real-trained models differently.

This experiment will reuse the two trained models from Experiment 1. The planned approach involves applying post-training INT8 quantization to both models, comparing performance degradation, and benchmarking runtime speed on Raspberry Pi 5 to determine if synthetic-trained models degrade differently from compression compared to real-trained models.

**Status**: Next steps - to be implemented after Experiment 1 completion.

---

### Experiment 3: Detection vs Classification on Raspberry Pi 5 (Coming Next)

**Goal**: Find the better trade-off between accuracy and FPS for edge deployment.

This experiment will compare object detection vs image classification for real-time CCTV analytics. The planned approach involves training both a YOLO detector and a classifier on the same dataset, then evaluating accuracy and FPS on Raspberry Pi 5 to determine the optimal approach for edge deployment.

**Status**: Next steps - to be implemented after Experiment 2 completion.

---

## 📊 Dataset Details

### Real Data Sources

- **YouTube CCTV Videos (FRAMES)**: 1,637 videos
- **Home CCTV Frames**: 11,722 frames
- **Company CCTV Frames**: 8,807 frames
- **Total Raw Frames**: ~22,000 frames

### Synthetic Data

- **50 Day Backgrounds**: Residential/commercial outdoor scenes (daytime)
- **50 Night Backgrounds**: Same scenes but nighttime/grayscale
- **~6,000 Generated Persons**: Placed on backgrounds using GenAI
- **Generation Tool**: NanoBanana (Google Gemini)
- **Background Preservation**: No alteration of backgrounds (critical for fair comparison)

### Dataset Splits (After Preprocessing)

After deduplication, clustering, and capping:

- **Train**: 1,795 images (830 day / 965 night = 46.2% day)
- **Val**: 664 images (264 day / 400 night = 39.8% day)
- **Test**: 660 images (260 day / 400 night = 39.4% day)
- **Total**: 3,119 images

### Preprocessing Steps

1. **Annotation**: Done first on all images using CVAT (Computer Vision Annotation Tool)
2. **Deduplication**: Cosine similarity thresholding using SSCD embeddings (threshold: 0.90)
3. **Scene Clustering**: SSCD-based clustering to group similar camera angles
4. **Capping**: Nighttime clusters capped to balance day/night distribution
5. **Train/Val/Test Split**: Per-cluster allocation to prevent data leakage

### Dataset Availability

**Important Note**: Due to privacy constraints, legal considerations, and data protection regulations, the real CCTV dataset cannot be publicly released. Real CCTV footage contains identifiable individuals and is subject to strict privacy requirements.

However, the **synthetic dataset** can be made available to the university for research purposes. The synthetic dataset contains fully generated CCTV-style images with synthetic persons, free from privacy concerns, and can be used for further research or model development.

---

## 🛠️ Environment Setup

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- Raspberry Pi 5 (for Experiment 2 & 3 deployment)

### Dependencies

Install the required packages:

```bash
# Core ML frameworks
pip install torch torchvision
pip install ultralytics

# Computer vision
pip install opencv-python pillow

# Embeddings and clustering
# For deduplication (SSCD model needs to be obtained separately)
pip install faiss-cpu  # or faiss-gpu if using GPU
pip install transformers

# Data processing
pip install pandas numpy
pip install scikit-learn

# Google GenAI (for synthetic data generation)
pip install google-genai python-dotenv

# Optional: Jupyter for notebooks
pip install jupyter
```

### Configuration Files

Update the config files in `configs/` to point to your dataset paths:

- `mydata_capped.yaml`: Real CCTV data configuration
- `mydata_synthetic.yaml`: Synthetic data configuration

Example configuration structure:

```yaml
path: /path/to/datasets/my_data
train: /path/to/train_capped.txt
val: /path/to/val_capped.txt
test: /path/to/test_capped.txt

nc: 1
names: ['person']
```

---

## 🔄 Pipeline Workflow

### 1. Data Collection

Collect CCTV footage from:

- YouTube videos
- Home CCTV systems
- Company CCTV systems

Extract frames and organize by source.

### 2. Annotation

**Annotate all images first** using CVAT (Computer Vision Annotation Tool):

- Annotate all collected images before any preprocessing
- Export annotations in YOLO format
- Place in `datasets/my_data/labels/`

### 3. Deduplication

Remove near-duplicate frames using cosine similarity:

```bash
python scripts/dedupe_by_threshold.py \
    --backend sscd \
    --model /path/to/sscd_model.pt \
    --out_dir /path/to/dedupe_results \
    --threshold 0.90
```

This script:

- Loads SSCD embeddings (cached)
- Uses FAISS range search for efficient duplicate detection
- Outputs `dedupe_keep.txt` and `dedupe_remove.txt`

### 4. Scene Clustering

Assign images to scene clusters using SSCD:

```bash
python scripts/assign_clusters.py \
    --backend sscd \
    --model /path/to/sscd_model.pt \
    --out_dir /path/to/clustering_results \
    --counts_only
```

This script:

- Assigns each image to the nearest scene cluster centroid
- Outputs `cluster_assignments.csv` and `cluster_counts.csv`

### 5. Capping and Train/Val/Test Split

Create leakage-free splits with day/night balance:

```bash
python scripts/split.py \
    --counts_csv /path/to/cluster_counts.csv \
    --assignments_csv /path/to/cluster_assignments.csv \
    --out_dir /path/to/output \
    --seed 42
```

This script:

- Applies capping to over-represented clusters (nighttime)
- Allocates clusters to train/val/test to prevent leakage
- Outputs `train_capped.txt`, `val_capped.txt`, `test_capped.txt`

### 6. Synthetic Data Generation (Optional)

If generating synthetic data:

#### Generate Backgrounds

```bash
python scripts/generate_synthetic_backgrounds.py
```

This creates 50 day + 50 night backgrounds.

#### Add Synthetic Persons

```bash
python scripts/fake_background_synthetic_person_placement.py
```

This adds synthetic persons to backgrounds using NanoBanana (Google Gemini).

---

## 🚂 Training Models

### Experiment 1: Real vs Synthetic Training

#### Train Real Data Model

Using the Jupyter notebook (`notebooks/Experiment_1.ipynb`) or Python:

```python
from ultralytics import YOLO

# Load pretrained COCO model
model = YOLO("yolo11n.pt")

# Train on real CCTV data
model.train(
    data="configs/mydata_capped.yaml",
    epochs=40,
    freeze=0,          # Fully unfrozen backbone
    lr0=2e-4,          # Stable learning rate
    imgsz=832,         # High resolution for small objects
    batch=16,
    name="real_unfrozen_e40"
)

# Evaluate on test set
metrics = model.val(
    data="configs/mydata_capped.yaml",
    split="test",
    plots=True,
    save_json=True
)

print(f"mAP@50: {metrics.box.map50}")
print(f"mAP@50-95: {metrics.box.map}")
print(f"Precision: {metrics.box.mp}")
print(f"Recall: {metrics.box.mr}")
```

#### Train Synthetic Data Model

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="configs/mydata_synthetic.yaml",
    epochs=40,
    freeze=0,
    lr0=2e-4,
    imgsz=832,
    batch=16,
    name="synthetic_unfrozen_e40"
)

# Evaluate on real test set (critical for fair comparison)
metrics = model.val(
    data="configs/mydata_capped.yaml",  # Real test set!
    split="test",
    plots=True,
    save_json=True
)
```

#### Training Configuration

Both models use identical hyperparameters for fairness:

- **Model**: YOLO11n (nano)
- **Image Size**: 832×832
- **Batch Size**: 16
- **Learning Rate**: 2e-4
- **Epochs**: 40
- **Backbone**: Unfrozen
- **Optimizer**: AdamW (auto)
- **Augmentation**: Default Ultralytics augmentations
- **Hardware**: RTX 4090 via Vast.ai

### Experiment 2: Quantization (Coming Next)

This experiment is planned as a next step. Detailed implementation instructions will be added once Experiment 2 begins.

---

## 🔄 Reproducing Experiments

### Experiment 1: Real vs Synthetic

1. **Prepare Data**

   **Important**: Annotate all images first using CVAT before running any preprocessing steps.
   
   ```bash
   # Step 1: Run deduplication using SSCD
   python scripts/dedupe_by_threshold.py \
       --backend sscd \
       --model /path/to/sscd_model.pt \
       --out_dir dedupe_results \
       --threshold 0.90

   # Step 2: Run clustering
   python scripts/assign_clusters.py \
       --backend sscd \
       --model /path/to/sscd_model.pt \
       --out_dir clustering_results \
       --counts_only

   # Step 3: Create splits (includes capping)
   python scripts/split.py \
       --counts_csv clustering_results/cluster_counts.csv \
       --assignments_csv clustering_results/cluster_assignments.csv \
       --out_dir data/
   ```
2. **Update Config Files**

   - Edit `configs/mydata_capped.yaml` with correct paths
   - Edit `configs/mydata_synthetic.yaml` with synthetic data paths
3. **Train Models**

   - Open `notebooks/Experiment_1.ipynb`
   - Run training cells for both real and synthetic models
   - Or use the Python scripts above
4. **Evaluate**

   - Both models evaluate on the same real test set
   - Compare metrics: mAP@50, mAP@50-95, Precision, Recall

### Experiment 2: Quantization (Coming Next)

Detailed reproduction instructions will be provided once Experiment 2 is implemented.

### Experiment 3: Detection vs Classification (Coming Next)

Detailed reproduction instructions will be provided once Experiment 3 is implemented.

---

## 📊 Results Summary

### Experiment 1: Real vs Synthetic (Completed)

#### Baseline (COCO Pretrained)

- **mAP@0.50**: 0.853
- **mAP@0.50:0.95**: 0.671
- **Precision**: 0.922
- **Recall**: 0.753

#### Real-Trained Model

- **Model**: `runs/detect/real_unfrozen_e50/weights/best.pt`
- **Training Data**: 1,795 real CCTV images
- **mAP@0.50**: 0.931
- **mAP@0.50:0.95**: 0.735
- **Precision**: 0.949
- **Recall**: 0.893

#### Synthetic-Trained Model

- **Model**: `runs/detect/synthetic_unfrozen_e40/weights/best.pt`
- **Training Data**: 1,795 synthetic CCTV images (matched to real dataset size)
- **mAP@0.50**: 0.719
- **mAP@0.50:0.95**: 0.503
- **Precision**: 0.769
- **Recall**: 0.615

#### Key Findings

- Real-trained model outperforms synthetic-only model
- Clear domain gap exists between synthetic and real CCTV scenes
- Synthetic data cannot fully replace real data, but may be useful for augmentation

---

## 🔮 Future Work

### Experiment 2: Quantization Impact

- [ ] Apply INT8 quantization to both models
- [ ] Measure accuracy degradation
- [ ] Benchmark FPS on Raspberry Pi 5
- [ ] Analyze quantization effects on synthetic vs real models

### Experiment 3: Detection vs Classification

- [ ] Train image classifier on same dataset
- [ ] Compare detection vs classification accuracy
- [ ] Benchmark FPS on Raspberry Pi 5
- [ ] Determine optimal architecture for edge deployment

### Additional Improvements

- [ ] Hybrid training: Combine real + synthetic data
- [ ] Domain adaptation techniques
- [ ] More diverse synthetic data generation
- [ ] Real-time inference optimization
- [ ] Deployment pipeline for Raspberry Pi 5

---

---

## 📝 License

This project was completed as part of a Final Year Project (FYP) at Technological University Dublin (TU Dublin).

Copyright © 2025 Tadhg Roche

All rights reserved. This work is the intellectual property of the author and was submitted as part of academic requirements. The synthetic dataset may be made available to the university for research purposes, but the real CCTV dataset cannot be shared due to privacy constraints.

## 👤 Author

**Tadhg Roche**

- Entrepreneur, Developer, IT Student, AI Engineer

---

## 🙏 Acknowledgments

- Ultralytics for the YOLO framework
- Vast.ai for GPU compute resources
- Open source community for tools and libraries

---

*Last updated: November 2025*
