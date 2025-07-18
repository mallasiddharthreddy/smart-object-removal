from PIL import Image
import numpy as np
from pathlib import Path
from predict_wrapper import run_lama_predict

# Paths
image_path = "input_images/000814.png"
mask_path = "input_images/000814_mask.png"
lama_model_dir = "lama/LaMa_models/big-lama"
output_dir = "predict_output"

# Load image and mask
image_pil = Image.open(image_path).convert("RGB")
mask_pil = Image.open(mask_path).convert("L")

# ✅ Save numpy arrays for debugging
np.save("test_input_array.npy", np.array(image_pil))
np.save("test_mask_array.npy", np.array(mask_pil))

# Run predict wrapper
result = run_lama_predict(
    image_pil,
    mask_pil,
    lama_model_dir=lama_model_dir,
    output_dir=output_dir,
    checkpoint_name="best.ckpt"
)

# Save and display result
result_path = f"{output_dir}/final_test_result.png"
result.save(result_path)
print(f"Inpainted image saved to: {result_path}")
