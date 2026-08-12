import os
import gc
import gradio as gr
from gradio import Server
from fastapi.responses import HTMLResponse
import numpy as np
import spaces
import torch
import random
import base64
import json
from io import BytesIO
from PIL import Image

from diffusers import Flux2KleinPipeline

MAX_SEED = np.iinfo(np.int32).max
LANCZOS = getattr(Image, "Resampling", Image).LANCZOS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

print("Using device:", device)

print("Loading FLUX.2 Klein 9B model base...")
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-9B",
    torch_dtype=torch.bfloat16,
).to(device)
print("Base Model loaded successfully.")

# ── LoRA adapter registry ──────────────────────────────────────────────────────
FACE_SWAP_PROMPT = """head_swap: start with Picture 1 as the base image, keeping its lighting, environment, and background. Remove the head from Picture 1 completely and replace it with the head from Picture 2.
FROM PICTURE 1 (strictly preserve):
- Scene: lighting conditions, shadows, highlights, color temperature, environment, background
- Head positioning: exact rotation angle, tilt, direction the head is facing
- Expression: facial expression, micro-expressions, eye gaze direction, mouth position, emotion
FROM PICTURE 2 (strictly preserve identity):
- Facial structure: face shape, bone structure, jawline, chin
- All facial features: eye color, eye shape, nose structure, lip shape and fullness, eyebrows
- Hair: color, style, texture, hairline
- Skin: texture, tone, complexion
The replaced head must seamlessly match Picture 1's lighting and expression while maintaining the complete identity from Picture 2. High quality, photorealistic, sharp details, 4k."""

ADAPTER_SPECS = {
    "None": {
        "repo": None,
        "weights": None,
        "adapter_name": None,
    },
    "Klein-Consistency": {
        "repo": "dx8152/Flux2-Klein-9B-Consistency",
        "weights": "Klein-consistency.safetensors",
        "adapter_name": "klein-consistency",
    },
    "Klein-Delight-Style": {
        "repo": "linoyts/Flux2-Klein-Delight-LoRA",
        "weights": "pytorch_lora_weights.safetensors",
        "adapter_name": "klein-delight",
    },
    "Best-Face-Swap": {
        "repo": "Alissonerdx/BFS-Best-Face-Swap",
        "weights": "bfs_head_v1_flux-klein_9b_step3750_rank64.safetensors",
        "adapter_name": "face-swap",
    },
    "Ghost-Mannequin": {
        "repo": "nhathoangfoto/FLUX.2-klein-ghost-mannequin",
        "weights": "3D-GhosMannequinRank-256_000005000.safetensors",
        "adapter_name": "ghost-mannequin",
    },
}

LOADED_ADAPTERS: set = set()
ADAPTER_NAMES = list(ADAPTER_SPECS.keys())

EXAMPLES_CONFIG = [
    {"images": ["examples/C.jpg"], "prompt": "Change the weather to stormy.", "lora": "Klein-Consistency"},
    {"images": ["examples/2.jpg"], "prompt": "Relight the image to remove all existing lighting conditions and replace them with neutral, uniform illumination. Apply soft, evenly distributed lighting with no directional shadows, no harsh highlights, and no dramatic contrast. Maintain the original identity of all subjects exactly—preserve facial structure, skin tone, proportions, expressions, hair, clothing, and textures. Do not alter pose, camera angle, background geometry, or image composition. Lighting should appear balanced, and studio-neutral, similar to diffuse overcast or a soft lightbox setup. Ensure consistent exposure across the entire image with realistic depth and subtle shading only where necessary for form.", "lora": "Klein-Delight-Style"},
    {"images": ["examples/1.jpg", "examples/2.jpg"], "prompt": FACE_SWAP_PROMPT, "lora": "Best-Face-Swap"},
    {"images": ["examples/1.jpg"], "prompt": "cinematic polaroid with soft grain subtle vignette gentle lighting white frame handwritten photographed by prithivMLmods preserving realistic texture and details", "lora": "None"},
    {"images": ["examples/cloth.jpg"], "prompt": "3Dghostmannequin", "lora": "Ghost-Mannequin"},
]


def make_thumb_b64(path, max_dim=220):
    if not os.path.exists(path):
        return ""
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_dim, max_dim), LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=65)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception as e:
        print(f"Thumbnail error for {path}: {e}")
        return ""


def encode_full_image(path):
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            data = f.read()
        ext = path.rsplit(".", 1)[-1].lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"
    except Exception as e:
        print(f"Encode error for {path}: {e}")
        return ""


def build_client_config():
    """Static config consumed by the frontend: LoRA list + example cards."""
    examples = []
    for i, ex in enumerate(EXAMPLES_CONFIG):
        examples.append({
            "idx": i,
            "thumbs": [make_thumb_b64(p) for p in ex["images"]],
            "n_images": len(ex["images"]),
            "lora": ex["lora"],
            "prompt": ex["prompt"],
        })
    return {
        "loras": ADAPTER_NAMES,
        "default_lora": "None",
        "examples": examples,
    }


print("Building client config (example thumbnails)…")
CLIENT_CONFIG = build_client_config()
print(f"Built config with {len(EXAMPLES_CONFIG)} examples and {len(ADAPTER_NAMES)} LoRAs.")


def b64_to_pil_list(b64_json_str):
    if not b64_json_str or b64_json_str.strip() in ("", "[]"):
        return []
    try:
        b64_list = json.loads(b64_json_str)
    except Exception:
        return []
    pil_images = []
    for b64_str in b64_list:
        if not b64_str or not isinstance(b64_str, str):
            continue
        try:
            if b64_str.startswith("data:image"):
                _, data = b64_str.split(",", 1)
            else:
                data = b64_str
            image_data = base64.b64decode(data)
            pil_images.append(Image.open(BytesIO(image_data)).convert("RGB"))
        except Exception as e:
            print(f"Error decoding image: {e}")
    return pil_images


def pil_to_b64_png(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def update_dimensions_on_upload(image):
    """Resizes image to be divisible by 16 to avoid tensor mismatch errors in FLUX."""
    if image is None:
        return 1024, 1024
    
    original_width, original_height = image.size
    
    scale = min(1024 / original_width, 1024 / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    new_width = (new_width // 16) * 16
    new_height = (new_height // 16) * 16
    
    return new_width, new_height


# ── Gradio Server (Server mode): FastAPI + Gradio queue/API engine ────────────
app = Server(title="FLUX.2-Klein-LoRA-Studio")


@app.mcp.tool(name="edit_image")
@app.api(name="edit_image")
@spaces.GPU(size="xlarge")
def infer(
    images_b64_json: str,
    prompt: str,
    lora_adapter: str,
    seed: int,
    randomize_seed: bool,
    guidance_scale: float,
    steps: int,
) -> dict:
    """Edit one or more images with FLUX.2 Klein + a lazily-loaded LoRA.

    Returns {"image": <base64 PNG data URL>, "seed": <seed used>}.
    """
    gc.collect()
    torch.cuda.empty_cache()

    pil_images = b64_to_pil_list(images_b64_json)
    if not pil_images:
        raise gr.Error("Please upload at least one image to edit.")
    if not prompt or prompt.strip() == "":
        raise gr.Error("Please enter an edit prompt.")

    spec = ADAPTER_SPECS.get(lora_adapter)
    if not spec:
        raise gr.Error(f"Configuration not found for: {lora_adapter}")

    # Face swap validation
    if spec["adapter_name"] == "face-swap":
        if len(pil_images) < 2:
            raise gr.Error("Face Swap requires exactly 2 images: Picture 1 (base/body) and Picture 2 (face source). Please upload 2 images.")
        elif len(pil_images) > 2:
            pil_images = pil_images[:2]

    if spec["adapter_name"] is None:
        print("Selection is None. Disabling LoRA adapters.")
        pipe.disable_lora()
    else:
        adapter_name = spec["adapter_name"]
        if adapter_name not in LOADED_ADAPTERS:
            print(f"--- Downloading and Loading Adapter: {lora_adapter} ---")
            try:
                pipe.load_lora_weights(spec["repo"], weight_name=spec["weights"], adapter_name=adapter_name)
                LOADED_ADAPTERS.add(adapter_name)
            except Exception as e:
                raise gr.Error(f"Failed to load adapter {lora_adapter}: {e}")
        else:
            print(f"--- Adapter {lora_adapter} already loaded. ---")

        print(f"Activating LoRA: {adapter_name}")
        pipe.set_adapters([adapter_name], adapter_weights=[1.0])

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Get dimensions from first image
    width, height = update_dimensions_on_upload(pil_images[0])
    
    # Process all images to the same dimensions
    processed_images = [
        img.resize((width, height), LANCZOS).convert("RGB") 
        for img in pil_images
    ]

    try:
        # Pass single image or list based on count
        image_input = processed_images if len(processed_images) > 1 else processed_images[0]
        
        result_image = pipe(
            image=image_input, 
            prompt=prompt,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]
        return {"image": pil_to_b64_png(result_image), "seed": seed}
    except Exception as e:
        raise e
    finally:
        gc.collect()
        torch.cuda.empty_cache()


@app.api(name="load_example", queue=False)
def load_example(idx: float) -> dict:
    """Return base64-encoded example images + prompt + LoRA for a given example index."""
    try:
        i = int(idx)
    except (ValueError, TypeError):
        i = -1
    if i < 0 or i >= len(EXAMPLES_CONFIG):
        return {"images": [], "prompt": "", "lora": "", "names": [], "status": "error"}
    ex = EXAMPLES_CONFIG[i]
    b64_list, names = [], []
    for path in ex["images"]:
        b64 = encode_full_image(path)
        if b64:
            b64_list.append(b64)
            names.append(os.path.basename(path))
    return {"images": b64_list, "prompt": ex["prompt"], "lora": ex["lora"], "names": names, "status": "ok"}


@app.get("/api/config")
def client_config():
    """Plain FastAPI route: LoRA choices + example card data for the frontend."""
    return CLIENT_CONFIG


@app.get("/", response_class=HTMLResponse)
async def homepage():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    app.launch(show_error=True, mcp_server=True)