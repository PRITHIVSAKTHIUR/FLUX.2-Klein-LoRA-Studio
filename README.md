# **[FLUX.2-Klein-LoRA-Studio](https://huggingface.co/spaces/prithivMLmods/FLUX.2-Klein-LoRA-Studio)**

FLUX.2-Klein-LoRA-Studio is an experimental, high-performance image-to-image manipulation and style-transfer platform built on top of the `black-forest-labs/FLUX.2-klein-9B` model. The application integrates specialized LoRA adapters—such as `Klein-Consistency`, `Klein-Delight-Style`, `Best-Face-Swap`, and `Ghost-Mannequin`—to deliver precision edits including consistency-preserved transformations, lighting neutralizations, identity-locked face swaps, and 3D apparel volume creation.

The workspace is served via a single-page web app built with a FastAPI backend server (`gradio.Server`) and a dark-mode frontend interface featuring a dual-view canvas, A/B comparison slider, history filmstrip, and interactive prompt suggestions.

<img width="1919" height="846" alt="Screenshot 2026-08-12 105139" src="https://github.com/user-attachments/assets/6284304c-217e-4b57-9ff3-970de3181df0" />

### **Key Features**

* **Lazy-Loaded Adapter Registry:** On-demand downloading and weight-fusing for specialized FLUX.2 Klein LoRA adapters (e.g., *Klein-Consistency*, *Klein-Delight-Style*, *Best-Face-Swap*, and *Ghost-Mannequin*).
* **Identity-Preserving Face Swap:** Incorporates a dual-input configuration (`Alissonerdx/BFS-Best-Face-Swap`) designed to seamlessly combine the background environment and micro-expressions of a base image with the target facial structure of a source portrait.
* **Text-Guided Image Editing:** Enables style transformations, camera angle shifts, lighting neutralization, and apparel volume generation in 4-step sampling windows.
* **Studio SPA Interface:** An interactive single-page application built with modern vanilla web components—featuring an A/B image comparison slider, history filmstrip, quick prompt chips, and drag-and-drop file support.
* **Smart Aspect Ratio Snapping:** Automatically resizes uploaded images to stay within 1024px while snapping width and height to multiples of 16 to prevent shape mismatch errors during inference.

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
├── qwenimage/
│   ├── __init__.py
│   ├── pipeline_qwenimage_edit_plus.py
│   ├── qwen_fa3_processor.py
│   └── transformer_qwenimage.py
├── app.py
├── index.html
├── LICENSE.txt
├── pre-requirements.txt
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock

```

### **Installation and Requirements**

To set up the FLUX.2-Klein-LoRA-Studio environment locally, configure your system according to the specifications below. A modern CUDA-enabled GPU is required.

* **Python Version:** Minimum Python **3.12** is needed; Python **3.12** or **3.14** is recommended.
* **PyTorch Version:** `torch==2.11.0` or above is required for best compatibility.
* **CUDA Version:** CUDA **13.0** is recommended (`--extra-index-url [https://download.pytorch.org/whl/cu130](https://download.pytorch.org/whl/cu130)`), matching the environment used on the live Hugging Face demo.

#### **Running with `uv` (Recommended)**

`uv` is an ultra-fast Python package and project manager written in Rust. It ensures rapid virtual environment setup and exact dependency synchronization based on the `uv.lock` file.

**Step 1 — Install `uv`**

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

#### **Standard PIP Implementation**

**1. Update Package Manager**
Upgrade your local package manager:

```bash
pip install pip>=26.1.2
```

**2. Install Core Dependencies**
Install the primary deep learning stack, transformer libraries, and core computing utilities listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

#### **Core Requirements List (`requirements.txt`)**

```text
--extra-index-url https://download.pytorch.org/whl/cu130
torch==2.11.0
torchvision==0.26.0
transformers==5.14.1
accelerate==1.14.0
diffusers==0.39.0
peft==0.19.1
gradio==6.22.0
av==17.1.0
spaces==0.51.1
huggingface-hub==1.24.0
kernels==0.16.0
```

### **Usage**

Once the web server initializes, open your browser to the local address output in your terminal (typically `http://127.0.0.1:7860/`).

1. **Upload Asset:** Drag and drop an image into the main canvas workspace, paste an image from your clipboard, or click the upload icon in the left rail.
* *Note:* Standard editing/consistency tasks use a single base image. Selecting **Best-Face-Swap** requires uploading **two** images: Image 1 acts as the base environment, and Image 2 acts as the facial identity source.
2. **Select Style / LoRA:** Choose your target editing task from the **Style / LoRA** dropdown menu in the right inspector panel. The adapter weights will download lazily on first use.
3. **Refine Instructions:** Type your instructions inside the prompt field, or click one of the **Quick Prompts** chips to instantly fill it. Press ⌘/Ctrl + Enter or click **Edit Image**.
4. **Compare & Chain:** Use the **Compare** tool on the left rail to view an A/B slider of the before and after states. Click **Use as Input** to chain multiple edits sequentially.

### **Links and Source**

* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/FLUX.2-Klein-LoRA-Studio](https://github.com/PRITHIVSAKTHIUR/FLUX.2-Klein-LoRA-Studio)
* **Hugging Face Live Space:** [https://huggingface.co/spaces/prithivMLmods/FLUX.2-Klein-LoRA-Studio](https://huggingface.co/spaces/prithivMLmods/FLUX.2-Klein-LoRA-Studio)
* **License:** [Apache License 2.0](https://github.com/PRITHIVSAKTHIUR/FLUX.2-Klein-LoRA-Studio/blob/main/LICENSE.txt)
