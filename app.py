import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import torch
import os
import cv2
from segment_anything import sam_model_registry, SamPredictor
from streamlit_drawable_canvas import st_canvas
from predict_wrapper import run_lama_predict
from io import BytesIO

lama_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "lama"))
os.environ["PYTHONPATH"] = lama_root + ":" + os.environ.get("PYTHONPATH", "")

import sys
if lama_root not in sys.path:
    sys.path.insert(0, lama_root)

# --- Streamlit Page Config ---
st.set_page_config(page_title="SAM + LaMa Inpainting", layout="wide")
st.title("🎯 SAM + LaMa Image Inpainting")

# --- Upload Section ---
st.header("1. Upload Your PNG Image")
uploaded_file = st.file_uploader("Upload a PNG image", type=["png"])

original_image = None
if uploaded_file:
    original_image = Image.open(uploaded_file).convert("RGB")
    width, height = original_image.size
    st.image(original_image, caption="Original Image", use_column_width=False)

# --- Bounding Box Drawing ---
st.header("2. Draw Bounding Box to Select Object")
bbox = None

if original_image:
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=original_image,
        update_streamlit=True,
        height=height,
        width=width,
        drawing_mode="rect",
        key="canvas",
    )

    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        last_box = canvas_result.json_data["objects"][-1]
        left = int(last_box["left"])
        top = int(last_box["top"])
        width_box = int(last_box["width"])
        height_box = int(last_box["height"])
        bbox = [left, top, left + width_box, top + height_box]
        st.markdown(f"**Bounding Box:** x={left}, y={top}, width={width_box}, height={height_box}")

# --- Inpainting Trigger ---
st.header("3. Mask & Inpaint")
if original_image and bbox:
    if st.button("Generate Mask & Inpaint"):
        st.info("Running SAM for mask and LaMa for inpainting...")

        # Convert to numpy for SAM
        image_np = np.array(original_image)

        # Load SAM model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam.to(device)
        predictor = SamPredictor(sam)
        predictor.set_image(image_np)

        input_box = np.array(bbox)[None, :]
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=False
        )

        # --- Step 1: Get binary mask from SAM
        mask = masks[0].astype(np.uint8) * 255

        # --- Step 2: Dilation + Blur + Final Soft Blur
        kernel = np.ones((25, 25), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        blurred = cv2.GaussianBlur(dilated, (17, 17), 0)
        _, binary = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)
        final_mask = cv2.GaussianBlur(binary, (3, 3), 0)

        mask_image = Image.fromarray(final_mask).convert("L")


        # --- Save .npy for debugging
        try:
            np.save(os.path.abspath("streamlit_input_array.npy"), np.array(original_image))
            np.save(os.path.abspath("streamlit_mask_array.npy"), final_mask)
            print("✅ .npy files saved successfully")
        except Exception as e:
            print(f"❌ Failed to save .npy files: {e}")

        # --- Optional: Debug overlay to visually verify
        overlay = np.array(original_image).copy()
        overlay[final_mask == 255] = [255, 0, 0]  # red where mask is
        Image.fromarray(overlay).save("debug_overlay.png")
   

        # Run LaMa
        lama_model_dir = os.path.abspath("lama/LaMa_models/big-lama")
        inpainted_image = run_lama_predict(
            image_pil=original_image,
            mask_pil=mask_image,
            lama_model_dir=lama_model_dir,
            output_dir="predict_output"
        )

        # --- Display Results ---
        st.markdown("### 4. Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(original_image, caption="Original")
        with col2:
            st.image(mask_image, caption="Mask")
        with col3:
            st.image(inpainted_image, caption="Inpainted")

        
        # Convert PIL image to PNG byte stream
        buffer = BytesIO()
        inpainted_image.save(buffer, format="PNG")
        buffer.seek(0)

        # --- Download ---
        st.download_button(
            label="Download Inpainted Image",
            data=buffer ,
            file_name="inpainted.png",
            mime="image/png"
        )

else:
    st.warning("Upload an image and draw a bounding box to continue.")

st.markdown("---")
st.caption("Built with ❤️ using SAM + LaMa + Streamlit")
