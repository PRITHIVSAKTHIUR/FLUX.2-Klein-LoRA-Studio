# **FLUX.2-Klein-LoRA-Studio**

FLUX.2-Klein-LoRA-Studio is an experimental, advanced image-to-image manipulation and style translation ecosystem. Built upon the robust `black-forest-labs/FLUX.2-klein-9B` distilled foundation model, this suite integrates a dynamic multi-LoRA switching framework pre-loaded with highly specialized workflow adapters.

The environment includes pipelines for `Klein-Consistency` structural cloning, `Klein-Delight-Style` lighting removal, `Ghost-Mannequin` 3D apparel volume generation, and a powerful, dual-image `Best-Face-Swap` mapping engine. It features dynamic aspect ratio tensor snapping, aggressive garbage collection routines, and direct, on-the-fly adapter weight fusion. Enclosed inside a tailored, responsive Orange Red interface theme, FLUX.2-Klein-LoRA-Studio offers creators a flexible, real-time playground for evaluating cutting-edge latent composition tasks.

<img width="1412" height="1559" alt="image" src="https://github.com/user-attachments/assets/24f3ba3d-da35-42f2-a5a3-56825b10dd8a" />

### **Key Features**

* **Dynamic Multi-LoRA Workspace:** Alternate seamlessly between specific image-editing adapters (Consistency, Delight, Ghost Mannequin, or Face-Swap) via an interactive macro grid gallery.
* **Identity-Preserving Face Swap:** Incorporates a robust dual-input configuration (`Alissonerdx/BFS-Best-Face-Swap`) designed to seamlessly combine the background environment and micro-expressions of a base image with the target facial structure of a source portrait.
* **Automatic Dimension Snapping:** Detects input gallery asset arrays and scales parameters smoothly so dimensions snap strictly to multiples of 16, preventing tensor shape mismatches in the deep layers of the FLUX architecture.
* **Unified ZeroGPU Workflow:** Features memory cleanup utilities (`gc.collect()` and `torch.cuda.empty_cache()`) coupled with an `xlarge` executor block to guarantee maximum stability during image-to-image inference.

### **Repository Structure**

```text
├── examples/
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── C.jpg
│   ├── cloth.jpg
│   ├── face-swap.jpg
│   ├── image.webp
│   ├── mc.png
│   └── Snow-Klein-consistency.png
├── app.py
├── LICENSE.txt
├── pre-requirements.txt
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock

```

### **Installation and Requirements**

To initialize the FLUX.2-Klein-LoRA-Studio workspace locally, configure a **Python 3.10** environment with the dependencies listed below. A modern CUDA-enabled GPU is required.

This repository specifically maps to **PyTorch 2.11.0 and CUDA cu13** architectures.

#### **Running with `uv` (Recommended)**

`uv` is an ultra-fast Python package and project manager written in Rust, ensuring rapid virtual environment synchronization and reproducible execution loops.

**Step 1 — Install `uv**`

* **macOS / Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
* **Windows:** `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

**Step 2 — Clone the repository**

```bash
git clone https://github.com/PRITHIVSAKTHIUR/FLUX.2-Klein-LoRA-Studio.git
cd FLUX.2-Klein-LoRA-Studio

```

**Step 3 — Initialize the project and install dependencies**

```bash
uv sync

```

**Step 4 — Run the script**

```bash
uv run app.py

```

#### **Standard PIP Installation**

**1. Upgrade Package Manager**
Ensure your local system installer is completely up-to-date:

```bash
pip install pip>=26.1

```

**2. Core Dependency Pull**
Install the primary deep learning stack, transformer layers, and pipeline architectures from your local `requirements.txt` file:

```bash
pip install -r requirements.txt

```

#### **Core Requirements List (`requirements.txt`)**

```text
git+https://github.com/huggingface/diffusers.git
transformers==5.9.0
huggingface_hub
sentencepiece
bitsandbytes
torchvision
accelerate
spaces
hf_xet
gradio==6.19.0
numpy
torch==2.11.0
peft
av

```

### **Usage**

Once the web deployment initializes, open your browser to the local address provided in your terminal output (typically `http://127.0.0.1:7860/`).

1. **Load Reference Media:** Upload your source images directly into the primary **Upload Images** gallery.
* *Note:* Standard editing/consistency tasks use a single base image. Selecting **Best-Face-Swap** requires exactly **two** images: Image 1 acts as the base environment, and Image 2 acts as the facial identity source.


2. **Select Style Preset:** Click on a stylistic adapter target card inside the **Edit Style Gallery** card block to hot-fuse the adapter weights.
3. **Refine Prompt:** Type or modify your edit instructions. Choosing the *Best-Face-Swap* macro dynamically loads a structured, contextual prompt template into the input area.
4. **Execute:** Click **Apply Style** to start the CUDA inference process and render your consistency-preserved output.

### **License and Source**

* **License:** [Apache License 2.0](https://github.com/PRITHIVSAKTHIUR/FLUX.2-Klein-LoRA-Studio/blob/main/LICENSE.txt)
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/FLUX.2-Klein-LoRA-Studio](https://github.com/PRITHIVSAKTHIUR/FLUX.2-Klein-LoRA-Studio)
