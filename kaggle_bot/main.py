import os
import sys
import json
import subprocess

print("---  LTX WORKER STARTED (CUSTOM PATH FIX) ---")

# 1. DEFINE TARGET FOLDER
# We install our libraries here to keep them separate
target_lib_path = "/kaggle/working/my_libs"
os.makedirs(target_lib_path, exist_ok=True)

# 2. FORCE INSTALL TO TARGET
print(f" Installing libraries to {target_lib_path}...")
try:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "-t", target_lib_path,           # <--- INSTALL HERE
        "diffusers>=0.36.0",             # Force new version
        "transformers", 
        "accelerate", 
        "sentencepiece", 
        "protobuf<5.0.0",                # Fix Google conflict
        "--upgrade", 
        "--no-warn-script-location"
    ])
    print(" Libraries installed!")
except Exception as e:
    print(f" Install warning: {e}")

# 3. MAGIC SWITCH: FORCE PYTHON TO LOOK HERE FIRST
# This makes Python ignore the old system libraries and use ours
if target_lib_path not in sys.path:
    sys.path.insert(0, target_lib_path)  # <--- INSERT AT TOP
print(f" Path Updated. Top entry: {sys.path[0]}")

# 4. NOW IMPORT
import torch
try:
    from diffusers import LTXVideoPipeline
    from diffusers.utils import export_to_video
    print(f" LTXPipeline Loaded Successfully!")
except ImportError as e:
    print(f" CRITICAL IMPORT ERROR: {e}")
    sys.exit(1)

# 5. LOAD CONFIG
try:
    with open("config.json", "r") as f:
        config = json.load(f)
        mode = config.get("model", "standard")
except:
    mode = "standard"

# 6. SELECT MODEL
base_path = "/kaggle/input/ltx-video-weights"
if mode == "fast_fp8":
    print(" MODE: Turbo (FP8)")
    model_file = "ltxv-2b-0.9.8-distilled-fp8.safetensors"
else:
    print(" MODE: Standard (High Quality)")
    model_file = "ltxv-2b-0.9.8-distilled.safetensors"

full_path = os.path.join(base_path, model_file)
print(f" Loading Model: {full_path}")

# 7. RUN PIPELINE
pipe = LTXVideoPipeline.from_single_file(
    full_path, 
    torch_dtype=torch.bfloat16
).to("cuda")

pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

# 8. GENERATE
try:
    with open("prompt.txt", "r") as f:
        prompt = f.read().strip()
except:
    prompt = "Test run"

print(f" Action: {prompt}")

steps = 8 if mode == "fast_fp8" else 25 

video = pipe(
    prompt=prompt,
    width=768,
    height=512,
    num_frames=121,
    num_inference_steps=steps,
    guidance_scale=3.0,
    fps=24
).frames[0]

output_file = "output.mp4"
export_to_video(video, output_file, fps=24)
print(" DONE")