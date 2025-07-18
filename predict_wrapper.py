import os
import shutil
import uuid
from PIL import Image
import subprocess
import traceback
import sys

def run_lama_predict(
    image_pil: Image.Image,
    mask_pil: Image.Image,
    lama_model_dir: str,
    output_dir: str = "predict_output",
    checkpoint_name: str = "best.ckpt",
):
    session_id = str(uuid.uuid4())[:8]
    base_dir = os.path.abspath(os.getcwd())
    lama_root = os.path.join(base_dir, "lama")

    temp_input_dir = os.path.join(base_dir, "lama_temp_inputs", session_id)
    output_dir_abs = os.path.abspath(os.path.join(base_dir, output_dir))
    lama_model_dir_abs = os.path.abspath(lama_model_dir)
    predict_script_path = os.path.join(lama_root, "bin", "predict.py")

   
    config_path = os.path.join(lama_model_dir_abs, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[ERROR] config.yaml not found at {config_path}")


    os.makedirs(temp_input_dir, exist_ok=True)
    if os.path.exists(output_dir_abs):
        shutil.rmtree(output_dir_abs)
    os.makedirs(output_dir_abs, exist_ok=True)

    
    image_path = os.path.join(temp_input_dir, "image.png")
    mask_path = os.path.join(temp_input_dir, "image_mask.png")
    image_pil.save(image_path)
    mask_pil.save(mask_path)

    # Save for debugging
    os.makedirs("debug", exist_ok=True)
    image_pil.save(os.path.join("debug", "last_input_image.png"))
    mask_pil.save(os.path.join("debug", "last_input_mask.png"))

    
    command = [
        sys.executable,  # Use current Python interpreter
        predict_script_path,
        f"model.path={lama_model_dir_abs}",
        f"model.checkpoint={checkpoint_name}",
        f"indir={temp_input_dir}",
        f"outdir={output_dir_abs}",
    ]

    # update PYTHONPATH so LaMa can import saicinpainting properly
    env = os.environ.copy()
    lama_python_path = lama_root + ":" + env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = lama_python_path

    try:
        print(f"\n[LaMa] Running command:\n{' '.join(command)}\n")
        result = subprocess.run(
            command,
            check=True,
            cwd=lama_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print("[LaMa STDOUT]:\n", result.stdout)
        print("[LaMa STDERR]:\n", result.stderr)

    except subprocess.CalledProcessError as e:
        print("\n[LaMa] Subprocess failed:")
        print("[STDOUT]:\n", e.stdout)
        print("[STDERR]:\n", e.stderr)
        raise RuntimeError(f"LaMa subprocess failed:\n{e.stderr.strip()}")

    # output image
    png_files = [f for f in os.listdir(output_dir_abs) if f.lower().endswith(".png")]
    if not png_files:
        raise FileNotFoundError(f"[ERROR] No output PNG found in {output_dir_abs}")

    inpaint_path = os.path.join(output_dir_abs, png_files[0])
    final_path = os.path.join(output_dir_abs, "final_result.png")

    if os.path.exists(final_path):
        os.remove(final_path)
    os.rename(inpaint_path, final_path)

    # Clean any other temp PNGs
    for f in os.listdir(output_dir_abs):
        if f.lower().endswith(".png") and f != "final_result.png":
            os.remove(os.path.join(output_dir_abs, f))

    # Clean temp input
    shutil.rmtree(temp_input_dir)

    return Image.open(final_path).convert("RGB")
