# CycleGAN Inference Pipeline

A complete CycleGAN image-to-image translation inference system with both **Gradio web UI** and **PyQt5 desktop** interfaces. Based on the [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) framework with a refactored, decoupled inference pipeline.

## ✨ Features

- **Decoupled inference pipeline** — Clean separation of model loading, image preprocessing, and postprocessing
- **CPU/GPU adaptive inference** — Automatically selects the best available device
- **Gradio Web UI** — Browser-based interactive interface for real-time image translation
- **PyQt5 Desktop App** — Native desktop application with drag-and-drop support
- **Multiple pretrained models** — Supports `horse2zebra` and `apple2orange` out of the box

## 📁 Project Structure

```
cyclegan-inference/
├── convert_tool.py          # Core inference pipeline (model loading + translation)
├── gradioui.py              # Gradio web interface
├── gradioui.ipynb           # Jupyter notebook version (for debugging)
├── gradioui_simple.ipynb    # Simplified notebook version
├── pyqt5.py                 # PyQt5 desktop application
└── model/                   # Pretrained model weights (download separately)
    ├── horse2zebra.pth
    └── apple2orange.pth
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision gradio Pillow
# For desktop app:
pip install PyQt5
```

### Download Pretrained Models

Download the pretrained `.pth` weights and place them in the `model/` directory:

| Model | Description | Download |
|---|---|---|
| `horse2zebra.pth` | Horse ↔ Zebra translation | [Download](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) |
| `apple2orange.pth` | Apple ↔ Orange translation | [Download](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) |

### Run Gradio Web UI

```bash
python gradioui.py
```

Open `http://localhost:7860` in your browser, upload an image, and see the translated result.

### Run PyQt5 Desktop App

```bash
python pyqt5.py
```

### Use as Python Module

```python
from convert_tool import CycleGANInference

# Load model
model = CycleGANInference(model_path="model/horse2zebra.pth")

# Translate image
result = model.translate("input.jpg")
result.save("output.jpg")
```

## 🔧 Core Pipeline (`convert_tool.py`)

The inference pipeline handles the complete workflow:

1. **Model Loading** — Loads the Generator network with pretrained weights, adapts to CPU/GPU
2. **Preprocessing** — Resizes input image, normalizes to `[-1, 1]` range, converts to tensor
3. **Inference** — Forward pass through the Generator with `torch.no_grad()`
4. **Postprocessing** — Denormalizes output tensor, converts back to PIL Image

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgements

- [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) — Original CycleGAN implementation
- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — CycleGAN paper
