import os
import gc
import gradio as gr
import numpy as np
import spaces
import torch
import random
from PIL import Image
from typing import Iterable

from diffusers import Flux2KleinPipeline
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

colors.orange_red = colors.Color(
    name="orange_red", c50="#FFF0E5", c100="#FFE0CC", c200="#FFC299", c300="#FFA366",
    c400="#FF8533", c500="#FF4500", c600="#E63E00", c700="#CC3700", c800="#B33000",
    c900="#992900", c950="#802200",
)

class OrangeRedTheme(Soft):
    def __init__(
        self, *, primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.orange_red,
        neutral_hue: colors.Color | str = colors.slate, text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue, secondary_hue=secondary_hue, neutral_hue=neutral_hue,
            text_size=text_size, font=font, font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600", block_border_width="3px",
            block_shadow="*shadow_drop_lg", button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px", color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

orange_red_theme = OrangeRedTheme()
MAX_SEED = np.iinfo(np.int32).max

LORA_STYLES = [
    {
        "image": "https://huggingface.co/spaces/prithivMLmods/FLUX.2-Klein-LoRA-Studio/resolve/main/examples/image.webp",
        "title": "None",
        "adapter_name": None,
        "repo": None,
        "weights": None
    },
    {
        "image": "https://huggingface.co/linoyts/Flux2-Klein-Delight-LoRA/resolve/main/image_3.png",
        "title": "Klein-Delight-Style",
        "adapter_name": "klein-delight",
        "repo": "linoyts/Flux2-Klein-Delight-LoRA",
        "weights": "pytorch_lora_weights.safetensors"
    },
]

LOADED_ADAPTERS = set()

print("Loading FLUX.2 Klein 9B model base...")
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-9B",
    torch_dtype=torch.bfloat16,
).to(device)
print("Base Model loaded successfully.")

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

def update_style_selection(evt: gr.SelectData):
    """Update selected style based on gallery click"""
    selected_style = LORA_STYLES[evt.index]
    
    if selected_style["title"] == "None":
        info_text = "### Selected: None (FLUX.2-klein-9B) ✅"
    else:
        info_text = f"### Selected: {selected_style['title']} ✅"
    
    return info_text, evt.index

@spaces.GPU
def infer(
    input_image, 
    prompt, 
    selected_style_index,
    seed=42, 
    randomize_seed=True, 
    guidance_scale=1.0, 
    steps=4, 
    progress=gr.Progress(track_tqdm=True)
):
    gc.collect()
    torch.cuda.empty_cache()

    if not input_image:
        raise gr.Error("Please upload an image to apply a style to.")

    if selected_style_index is None:
        selected_style_index = 0
    
    selected_style = LORA_STYLES[selected_style_index]
    
    if selected_style["adapter_name"] is None:
        print("Selection is None. Disabling LoRA adapters.")
        pipe.disable_lora()
    else:
        adapter_name = selected_style["adapter_name"]
        
        if adapter_name not in LOADED_ADAPTERS:
            print(f"--- Downloading and Loading Adapter: {selected_style['title']} ---")
            try:
                pipe.load_lora_weights(
                    selected_style["repo"], 
                    weight_name=selected_style["weights"], 
                    adapter_name=adapter_name
                )
                LOADED_ADAPTERS.add(adapter_name)
            except Exception as e:
                raise gr.Error(f"Failed to load adapter {selected_style['title']}: {e}")
        else:
            print(f"--- Adapter {selected_style['title']} is already loaded. ---")
            
        print(f"Activating LoRA: {adapter_name}")
        pipe.set_adapters([adapter_name], adapter_weights=[1.0])

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    
    width, height = update_dimensions_on_upload(input_image)
    processed_input = input_image.resize((width, height), Image.LANCZOS).convert("RGB")
    
    try:
        image = pipe(
            image=processed_input, 
            prompt=prompt,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            num_inference_steps=steps,
            generator=torch.Generator(device=device).manual_seed(seed),
        ).images[0]
        
        return image, seed

    except Exception as e:
        raise gr.Error(f"Inference failed: {e}")
    finally:
        gc.collect()
        torch.cuda.empty_cache()

@spaces.GPU
def infer_example(input_image, prompt, style_index):
    if input_image is None: 
        return None, 0
        
    image, seed = infer(
        input_image=input_image, 
        prompt=prompt, 
        selected_style_index=style_index, 
        seed=0, 
        randomize_seed=True,
        guidance_scale=1.0,
        steps=4
    )
    return image, seed

css="""
#col-container { margin: 0 auto; max-width: 960px; }
#main-title h1 { font-size: 2.2em !important; }
#style_gallery .grid-wrap{height: 10vh}
"""

with gr.Blocks() as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# **FLUX.2-Klein-LoRA-Studio**", elem_id="main-title")
        gr.Markdown("Perform diverse image edits using specialized [LoRAs](https://huggingface.co/models?other=base_model:adapter:black-forest-labs/FLUX.2-klein-9B) adapters for the [FLUX.2-Klein-Distilled](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) model.")
        
        selected_style_index = gr.State(0)
        
        with gr.Row(equal_height=True):
            with gr.Column():
                input_image = gr.Image(
                    label="Upload Image", 
                    type="pil", 
                    height=290, 
                    sources=["upload", "webcam", "clipboard"]
                )
                with gr.Row():
                    prompt = gr.Text(
                        label="Edit Prompt", 
                        max_lines=1,
                        show_label=True, 
                        placeholder="e.g., a man with a red superhero mask"
                    )
                    
                run_button = gr.Button("Apply Style", variant="primary")

                with gr.Accordion("Advanced Settings", open=False, visible=False):
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    guidance_scale = gr.Slider(label="Guidance Scale", minimum=0.0, maximum=10.0, step=0.1, value=1.0)       
                    steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=4, step=1)
                    
            with gr.Column():
                output_image = gr.Image(label="Output Image", interactive=False, format="png", height=358)
                used_seed = gr.Textbox(label="Used Seed", interactive=False, visible=False)

        selected_style_info = gr.Markdown("### Selected: None (FLUX.2-klein-9B) ✅")
        
        style_gallery = gr.Gallery(
            [(item["image"], item["title"]) for item in LORA_STYLES],
            label="Edit Style Gallery",
            allow_preview=False,
            columns=2,
            elem_id="style_gallery",
        )
                    
        gr.Examples(
            examples=[
                ["examples/2.jpg", "Relight the image to remove all existing lighting conditions and replace them with neutral, uniform illumination. Apply soft, evenly distributed lighting with no directional shadows, no harsh highlights, and no dramatic contrast. Maintain the original identity of all subjects exactly—preserve facial structure, skin tone, proportions, expressions, hair, clothing, and textures. Do not alter pose, camera angle, background geometry, or image composition. Lighting should appear balanced, and studio-neutral, similar to diffuse overcast or a soft lightbox setup. Ensure consistent exposure across the entire image with realistic depth and subtle shading only where necessary for form."],
                ["examples/1.jpg", "cinematic polaroid with soft grain subtle vignette gentle lighting white frame handwritten photographed by prithivMLmods preserving realistic texture and details"], 
            ],
            inputs=[input_image, prompt],
            outputs=[output_image, used_seed],
            fn=infer_example,
            cache_examples=False,
        )
        
        gr.Markdown("[*](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)This is still an experimental Space for FLUX.2-Klein-9B. More adapters will be added soon.")
        
    style_gallery.select(
        update_style_selection,
        outputs=[selected_style_info, selected_style_index]
    )
    
    run_button.click(
        fn=infer,
        inputs=[input_image, prompt, selected_style_index, seed, randomize_seed, guidance_scale, steps],
        outputs=[output_image, used_seed]
    )

if __name__ == "__main__":
    demo.queue().launch(theme=orange_red_theme, css=css, show_error=True)