# Bounding Box-Based Image Segmentation using Segment Anything Model (SAM)

This repository demonstrates how to use the **Segment Anything Model (SAM)** for bounding box-based image segmentation using **PyQt5**.

## Features
- Uses **SAM (Segment Anything Model)** for segmentation
- Implements a **PyQt5 GUI** to load and segment images
- Provides options for bounding box-based selection

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/your-repository.git
cd your-repository
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
Ensure you have `PyQt5`, `torch`, `torchvision`, `matplotlib`, and `numpy` installed.

### 3. Download the SAM Model
To use the SAM model, download the checkpoint from Meta's official repository:
```bash
mkdir models
cd models
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
Alternatively, if `wget` is not available, use:
```bash
curl -o sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Usage
Run the PyQt5-based GUI:
```bash
python app.py
```

## Code Overview
The implementation consists of:
- **`sam_segmentation.py`**: Contains the SAM model loading and segmentation logic.
- **`gui.py`**: PyQt5 GUI to load images and apply segmentation.
- **`app.py`**: Main script to launch the application.

## How SAM Model is Loaded
```python
from segment_anything import SamPredictor, sam_model_registry
import torch

# Load the SAM model
sam_checkpoint = "models/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
predictor = SamPredictor(sam)
```
This initializes and loads the SAM model into memory for segmentation.

## Contributing
Feel free to submit issues or pull requests if you want to improve this project!

## License
[MIT License](LICENSE)

