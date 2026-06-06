# ARCE-YOLO: Enhancing small fruit detection with attention-guided receptive fields network during thinning period

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.8.0-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green)](LICENSE)

This repository contains the official implementation of **ARCE-YOLO**, proposed in our paper submitted to *Horticulturae*.

## Overview

ARCE-YOLO is a lightweight attention-guided object detection network built upon YOLOv11-n, specifically designed for small green fruit detection in natural orchard environments. Our method introduces three key modules:

- **AGLE** (Attention-Guided Lightweight Extraction): A downsampling module that uses CBAM attention to guide spatial information aggregation, reducing parameters while maintaining feature quality.
- **RFCBAMConv** (Receptive Field Convolutional Block Attention Module): A convolution module combining receptive field attention with channel/spatial attention for enhanced multi-scale feature extraction.
- **EUCB** (Efficient Upsampling Convolution Block): A lightweight upsampling module using depth-wise separable convolutions and channel shuffling for efficient feature fusion.

Additionally, we incorporate a **P2 detection layer** to leverage high-resolution feature maps for improved small object detection.

## Architecture

```
Input (640x640)
    |
    v
Backbone (YOLO11-n with AGLE + RFCBAMConv)
    |-- Conv (64, k=3, s=2)
    |-- Conv (128, k=3, s=2)
    |-- C3k2_RFCBAMConv x2
    |-- AGLE (P3/8)
    |-- C3k2_RFCBAMConv x2
    |-- AGLE (P4/16)
    |-- C3k2_RFCBAMConv x2
    |-- AGLE (P5/32)
    |-- C3k2_RFCBAMConv x2
    |-- SPPF
    |-- C2PSA
    |
    v
Neck (BiFPN-like with EUCB + AGLE)
    |-- EUCB + Concat + C3k2_RFCBAMConv (P4 fusion)
    |-- EUCB + Concat + C3k2_RFCBAMConv (P3 fusion)
    |-- EUCB + Concat + C3k2_RFCBAMConv (P2 fusion)
    |-- AGLE + Concat + C3k2_RFCBAMConv (P3 re-fusion)
    |-- AGLE + Concat + C3k2_RFCBAMConv (P4 re-fusion)
    |-- AGLE + Concat + C3k2_RFCBAMConv (P5 re-fusion)
    |
    v
Head: Detect(P2, P3, P4, P5)
```

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.8.0
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/[your-username]/ARCE-YOLO.git
cd ARCE-YOLO

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

### Training

```python
from ultralytics import YOLO

# Load ARCE-YOLO model
model = YOLO('ultralytics/cfg/models/11/yolo11_ARCE.yaml')

# Train on your dataset
model.train(
    data='your_dataset.yaml',  # Path to dataset configuration
    epochs=150,
    imgsz=640,
    batch=4,
    save_json=True,
    pretrained=False,
    optimizer='SGD',
    lr0=0.01,
    momentum=0.937,
    device=0  # GPU device ID, or 'cpu'
)
```

### Validation

```python
# Validate on test set
metrics = model.val(data='your_dataset.yaml', split='test')
print(f"mAP@0.5: {metrics.box.map50:.3f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")
```

### Inference

```python
# Run inference on an image
results = model('path/to/image.jpg')
results[0].show()
```

## Model Zoo

| Model | Parameters | mAP@0.5 | mAP@0.5:0.95 | APs |
|-------|-----------|---------|--------------|-----|
| YOLOv11-n (baseline) | 2.5M | 87.9% | 68.9% | 47.4% |
| **ARCE-YOLO (ours)** | **2.3M** | **90.9%** | **72.3%** | **54.4%** |

*Results on Golden Pear test set (2,438 images, 7:3 train-val split).*

## Datasets

### Golden Pear Dataset
Our private Golden Pear dataset contains 2,438 images of green pears in natural orchard environments, with bounding box annotations. Due to ongoing research and the significant investment in data collection and annotation, the complete dataset is not publicly released at this time.

**However, we provide the following alternatives:** The full dataset can be shared privately upon reasonable request.

### MinneApple Dataset
We also validate our method on the publicly available [MinneApple dataset](https://github.com/nicolaihaeni/MinneApple), which can be used to independently reproduce our results.

## Project Structure

```
ARCE-YOLO/
├── ultralytics/
│   ├── nn/
│   │   ├── Addmodules/          # Custom modules
│   │   │   ├── AGLE.py          # Attention-Guided Lightweight Extraction
│   │   │   ├── RFCBAMConv.py    # Receptive Field CBAM Convolution
│   │   │   ├── EUCB.py          # Efficient Upsampling Convolution Block
│   │   │   └── __init__.py
│   │   ├── modules/             # Base ultralytics modules
│   │   └── tasks.py             # Model registration (includes Addmodules)
│   ├── cfg/
│   │   └── models/
│   │       └── 11/
│   │           └── yolo11_ARCE.yaml  # ARCE-YOLO model config
│   ├── data/                    # Data loading and augmentation
│   ├── engine/                  # Training/validation engine
│   └── models/                  # Model definitions
├── requirements.txt
├── README.md
└── LICENSE
```

## Key Modules

### AGLE (Attention-Guided Lightweight Extraction)
Located in `ultralytics/nn/Addmodules/AGLE.py`

AGLE replaces standard strided convolutions for downsampling. It uses CBAM (Convolutional Block Attention Module) to generate spatial attention weights, which guide the aggregation of 2x2 spatial patches into a single feature. This reduces parameters by ~20% compared to standard downsampling while preserving critical spatial information.

### RFCBAMConv (Receptive Field Convolutional Block Attention Module)
Located in `ultralytics/nn/Addmodules/RFCBAMConv.py`

RFCBAMConv enhances the standard convolution by:
1. Generating multi-scale receptive field features through depth-wise convolutions
2. Applying channel attention (SE module) to emphasize informative channels
3. Applying spatial attention to focus on relevant spatial regions

It is used to replace standard convolutions in the C3k2 blocks throughout the backbone and neck.

### EUCB (Efficient Upsampling Convolution Block)
Located in `ultralytics/nn/Addmodules/EUCB.py`

EUCB replaces the standard ConvTranspose for upsampling in the feature fusion path. It uses:
1. Bilinear upsampling followed by depth-wise convolution
2. Channel shuffling for efficient information mixing
3. Point-wise convolution for channel adjustment

This reduces computational cost while maintaining upsampling quality.



## License

This project is licensed under the AGPL-3.0 License. See [LICENSE](LICENSE) for details.

## Acknowledgements

This work is built upon the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) framework. We thank the Ultralytics team for their excellent open-source contribution to the computer vision community.

## Contact

For questions or dataset access requests, please contact [your-email] or open an issue in this repository.
