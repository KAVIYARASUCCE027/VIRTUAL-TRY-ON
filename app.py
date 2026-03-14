import gradio as gr
import torch
from PIL import Image
import numpy as np
import socket
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def log(msg: str) -> None:
    print(msg, flush=True)
    logging.info(msg)

def find_free_port(start: int = 7860, end: int = 7900) -> int:
    for port in range(start, end):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("0.0.0.0", port))
                return port
        except OSError:
            continue
    return start

# Detect device
if torch.cuda.is_available():
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    log(f"Using device: CUDA ({gpu_name})")
else:
    device = "cpu"
    log("Using device: CPU (torch.cuda.is_available() == False)")

pipe = None

def get_pipe():
    global pipe
    if pipe is None:
        from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler

        log("Loading StableDiffusionInpaintPipeline from Hugging Face...")
        start_time = time.time()

        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        pipe_local = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch_dtype,
            safety_checker=None
        )

        pipe_local = pipe_local.to(device)

        # Memory optimizations
        pipe_local.enable_attention_slicing()

        if device == "cuda":
            try:
                pipe_local.enable_xformers_memory_efficient_attention()
                log("Enabled xFormers memory efficient attention")
            except:
                log("xFormers not available")

        pipe_local.scheduler = DPMSolverMultistepScheduler.from_config(pipe_local.scheduler.config)

        load_time = time.time() - start_time
        log(f"Model loaded in {load_time:.1f}s")

        pipe = pipe_local

    return pipe


def virtual_tryon(person_image, dress_image):
    log("virtual_tryon called")

    if person_image is None or dress_image is None:
        log("Missing input images")
        return None

    target_w, target_h = 512, 384
    person = person_image.resize((target_w, target_h))

    # Create mask
    mask = Image.new("L", (target_w, target_h), 0)
    mask_np = np.array(mask)

    y1, y2 = int(target_h * 0.3), int(target_h * 0.95)
    x1, x2 = int(target_w * 0.25), int(target_w * 0.75)

    mask_np[y1:y2, x1:x2] = 255
    mask = Image.fromarray(mask_np)

    prompt = "same person wearing the given dress, fashion photography, realistic fabric, preserve face"
    negative_prompt = "blurry, distorted body, extra limbs"

    pipeline = get_pipe()

    steps = 12 if device == "cuda" else 8

    log(f"Starting diffusion inference on {device} with {steps} steps")

    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=person,
        mask_image=mask,
        guidance_scale=7.5,
        num_inference_steps=steps
    ).images[0]

    log("Inference completed")

    return result


demo = gr.Interface(
    fn=virtual_tryon,
    inputs=[
        gr.Image(type="pil", label="Upload Person Image"),
        gr.Image(type="pil", label="Upload Dress Image"),
    ],
    outputs=gr.Image(label="Try-On Result"),
    title="👗 Virtual Fashion AI",
    description="Upload a person image and a dress image to generate a virtual try-on result."
)

port = find_free_port()

log(f"Launching server on port {port}")

demo.launch(
    server_name="0.0.0.0",
    server_port=port,
    share=True
)