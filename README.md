# CycleGAN Inference Project

A simplified inference system for CycleGAN, featuring a **Gradio web interface** running directly in a Jupyter Notebook. This project uses the official [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) implementation as its backend.

## 🚀 Quick Start

### 1. Install Dependencies

Ensure you have Python installed, then install the required libraries:

```bash
pip install torch torchvision gradio pillow
```

*(Note: If you have a GPU, make sure to install the CUDA version of PyTorch for better performance.)*

### 2. Download Models

Download the pretrained model weights (`.pth` files) and place them in the `model/` directory:

| Model | Task | Download Link |
|---|---|---|
| `horse2zebra.pth` | Horse ↔ Zebra | [Download](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) |
| `apple2orange.pth` | Apple ↔ Orange | [Download](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) |

### 3. Run the Demo

Open the Jupyter Notebook:

```bash
jupyter notebook gradioui_simple.ipynb
```

Run all cells in the notebook. A Gradio interface will appear at the bottom, where you can upload images and see the style transfer results in real-time.

## 📁 Project Structure

```
cyclegan-inference/
├── gradioui_simple.ipynb   # ✅ Main entry point (Run this!)
├── 官方源码/               # Official CycleGAN source code (Backend)
├── model/                  # Place your .pth model weights here
└── README.md
```

## 📄 How it Works

The notebook `gradioui_simple.ipynb` dynamically imports the network definitions from the `官方源码` folder, loads the pretrained weights, and wraps the inference pipeline in a simple Gradio UI.
