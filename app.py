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
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    return start


if torch.cuda.is_available():
    device = "cuda"
    try:
        gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        gpu_name = "Unknown GPU"
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
        t0 = time.time()
        try:
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
            pipe_local = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch_dtype,
                safety_checker=None,  # disable NSFW safety checker
            )
        except Exception as e:
            log(f"Error while loading model from Hugging Face: {e}")
            raise

        # Move pipeline to the selected device (GPU or CPU)
        pipe_local = pipe_local.to(device)

        # Memory optimizations for 4GB-class GPUs (safe on CPU too)
        pipe_local.enable_attention_slicing()
        if device == "cuda":
            try:
                pipe_local.enable_xformers_memory_efficient_attention()
                log("Enabled xFormers memory efficient attention.")
            except Exception:
                log("xFormers not available; using standard attention.")

        pipe_local.scheduler = DPMSolverMultistepScheduler.from_config(pipe_local.scheduler.config)

        load_time = time.time() - t0
        log(f"Model loaded and moved to {device} in {load_time:.1f}s.")

        if device == "cpu":
            log("Running on CPU: inference will be slower. Resolution and steps are reduced.")

        pipe = pipe_local
    return pipe


def virtual_tryon(person_image, dress_image):
    start_time = time.time()
    log("virtual_tryon called.")

    if person_image is None or dress_image is None:
        log("Input check failed: person_image or dress_image is None.")
        return None

    target_w, target_h = 512, 384
    log(f"Resizing input person image to {target_w}x{target_h}.")
    person = person_image.resize((target_w, target_h))

    log("Creating inpainting mask.")
    mask = Image.new("L", (target_w, target_h), 0)
    mask_np = np.array(mask)
    y1, y2 = int(target_h * 0.3), int(target_h * 0.95)
    x1, x2 = int(target_w * 0.25), int(target_w * 0.75)
    mask_np[y1:y2, x1:x2] = 255
    mask = Image.fromarray(mask_np)

    prompt = "same person wearing the given dress, preserve face, realistic fabric, fashion photography"
    negative_prompt = "distorted face, blurry, extra arms"

    pipeline = get_pipe()

    if device == "cuda":
        steps = 12
    else:
        steps = 8
    log(f"Starting diffusion inference on {device} with {steps} steps.")

    def progress_callback(step, timestep, latents):
        if step % 2 == 0:
            log(f"Inference progress: step {step}, timestep {timestep}")

    try:
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=person,
            mask_image=mask,
            guidance_scale=7.5,
            num_inference_steps=steps,
            callback=progress_callback,
            callback_steps=1,
        ).images[0]
    except Exception as e:
        log(f"Error during diffusion inference: {e}")
        raise

    total_time = time.time() - start_time
    log(f"Inference completed in {total_time:.1f}s.")

    return result


demo = gr.Interface(
    fn=virtual_tryon,
    inputs=[
        gr.Image(type="pil", label="Upload Person Image"),
        gr.Image(type="pil", label="Upload Dress Image"),
    ],
    outputs=gr.Image(label="Try-On Result"),
    title="👗 Virtual Fashion AI",
    description="Upload a person and a dress image to generate virtual try-on.",
)

port = find_free_port()
log(f"Using port {port} — open http://127.0.0.1:{port}")
demo.launch(server_name="127.0.0.1", server_port=port)
