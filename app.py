import os
import json
import math
import numpy as np
import cv2
import torch
import pandas as pd
import gradio as gr
import joblib
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter, ImageChops
from transformers import AutoImageProcessor, AutoModelForImageClassification
import concurrent.futures

# ==========================================================
# 1. LOAD MODEL AND FEATURES
# ==========================================================

# Make sure these two files are in the same folder as this script!
MODEL_PATH = "ensemble_v2.joblib"
FEATURES_PATH = "features_v2.json"

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    print("WARNING: ensemble_v2.joblib or features_v2.json not found in the current directory.")
    print("Please make sure you have downloaded both from your Kaggle models_v2 folder.")

ensemble = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

if os.path.exists(FEATURES_PATH):
    with open(FEATURES_PATH, "r") as f:
        FEATURE_NAMES = json.load(f)
else:
    FEATURE_NAMES = []

# ==========================================================
# 2. TRANSFORMATIONS
# ==========================================================

def jpeg_compress(img, quality):
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    result = Image.open(buffer).convert("RGB")
    result.load()
    return result

def gaussian_blur(img, radius):
    return img.filter(ImageFilter.GaussianBlur(radius))

def sharpen(img, factor):
    return ImageEnhance.Sharpness(img).enhance(factor)

def brightness(img, factor):
    return ImageEnhance.Brightness(img).enhance(factor)

def contrast(img, factor):
    return ImageEnhance.Contrast(img).enhance(factor)

def gaussian_noise(img, sigma):
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def rotate(img, angle):
    return img.rotate(angle, expand=False, fillcolor=(128, 128, 128))

def horizontal_flip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def hue_shift(img, shift):
    hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + shift) % 180
    return Image.fromarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))

def saturation(img, factor):
    return ImageEnhance.Color(img).enhance(factor)

def resize_scale(img, scale):
    w, h = img.size
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))))

def center_crop(img, percent):
    w, h = img.size
    nw = int(w * percent)
    nh = int(h * percent)
    left = (w - nw) // 2
    top = (h - nh) // 2
    return img.crop((left, top, left + nw, top + nh))

def screenshot_phone(img):
    w, h = img.size
    new_h = int(h * (1080 / max(1, w)))
    img = img.resize((1080, max(1, new_h)))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    result = Image.open(buffer).convert("RGB")
    result.load()
    return result

def screenshot_social(img):
    w, h = img.size
    new_h = int(h * (1080 / max(1, w)))
    img = img.resize((1080, max(1, new_h)))
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    result = Image.open(buffer).convert("RGB")
    result.load()
    return result

def screenshot_messaging(img):
    w, h = img.size
    new_h = int(h * (720 / max(1, w)))
    img = img.resize((720, max(1, new_h)))
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=70)
    buffer.seek(0)
    result = Image.open(buffer).convert("RGB")
    result.load()
    return result

transformations = {
    "none":           lambda x: x,
    "jpeg_90":        lambda x: jpeg_compress(x, 90),
    "jpeg_70":        lambda x: jpeg_compress(x, 70),
    "jpeg_50":        lambda x: jpeg_compress(x, 50),
    "blur_2":         lambda x: gaussian_blur(x, 2),
    "blur_4":         lambda x: gaussian_blur(x, 4),
    "blur_6":         lambda x: gaussian_blur(x, 6),
    "sharp_1.5":      lambda x: sharpen(x, 1.5),
    "sharp_2":        lambda x: sharpen(x, 2),
    "sharp_3":        lambda x: sharpen(x, 3),
    "bright_0.7":     lambda x: brightness(x, 0.7),
    "bright_1.3":     lambda x: brightness(x, 1.3),
    "bright_1.6":     lambda x: brightness(x, 1.6),
    "contrast_0.7":   lambda x: contrast(x, 0.7),
    "contrast_1.3":   lambda x: contrast(x, 1.3),
    "contrast_1.6":   lambda x: contrast(x, 1.6),
    "noise_5":        lambda x: gaussian_noise(x, 5),
    "noise_15":       lambda x: gaussian_noise(x, 15),
    "noise_30":       lambda x: gaussian_noise(x, 30),
    "rotate_5":       lambda x: rotate(x, 5),
    "rotate_15":      lambda x: rotate(x, 15),
    "rotate_30":      lambda x: rotate(x, 30),
    "flip":           lambda x: horizontal_flip(x),
    "hue_10":         lambda x: hue_shift(x, 10),
    "hue_30":         lambda x: hue_shift(x, 30),
    "hue_60":         lambda x: hue_shift(x, 60),
    "sat_0.7":        lambda x: saturation(x, 0.7),
    "sat_1.3":        lambda x: saturation(x, 1.3),
    "sat_1.8":        lambda x: saturation(x, 1.8),
    "resize_75":      lambda x: resize_scale(x, 0.75),
    "resize_50":      lambda x: resize_scale(x, 0.50),
    "resize_25":      lambda x: resize_scale(x, 0.25),
    "crop_95":        lambda x: center_crop(x, 0.95),
    "crop_85":        lambda x: center_crop(x, 0.85),
    "crop_70":        lambda x: center_crop(x, 0.70),
    "screenshot_phone":     lambda x: screenshot_phone(x),
    "screenshot_social":    lambda x: screenshot_social(x),
    "screenshot_messaging": lambda x: screenshot_messaging(x),
}

# ==========================================================
# 3. HUGGING FACE MODELS (SigLIP & ViT)
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SigLIP
siglip_model_name = "Ateeqq/ai-vs-human-image-detector"
siglip_processor = AutoImageProcessor.from_pretrained(siglip_model_name)
siglip_model = AutoModelForImageClassification.from_pretrained(siglip_model_name).to(device)

_siglip_id2label = siglip_model.config.id2label
_siglip_ai_idx = 1
for idx, label in _siglip_id2label.items():
    if any(kw in str(label).lower() for kw in ["ai", "fake", "generated", "artificial"]):
        _siglip_ai_idx = int(idx)
        break

def siglip_batch_detector(images_list):
    if not images_list: return []
    inputs = siglip_processor(images=images_list, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = siglip_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    
    results = []
    for i in range(len(images_list)):
        ai_prob = float(probs[i, _siglip_ai_idx])
        results.append({
            "siglip_ai_prob": ai_prob,
            "siglip_confidence": float(probs[i].max()),
        })
    return results

# ViT
vit_model_name = "dima806/ai_vs_human_generated_image_detection"
vit_processor = AutoImageProcessor.from_pretrained(vit_model_name)
vit_model = AutoModelForImageClassification.from_pretrained(vit_model_name).to(device)

_vit_id2label = vit_model.config.id2label
_vit_ai_idx = 1
for idx, label in _vit_id2label.items():
    if any(kw in str(label).lower() for kw in ["ai", "fake", "generated", "artificial"]):
        _vit_ai_idx = int(idx)
        break

def vit_batch_detector(images_list):
    if not images_list: return []
    inputs = vit_processor(images=images_list, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = vit_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    
    results = []
    for i in range(len(images_list)):
        ai_prob = float(probs[i, _vit_ai_idx])
        results.append({
            "vit_ai_prob": ai_prob,
            "vit_confidence": float(probs[i].max()),
        })
    return results

# ==========================================================
# 4. CPU DETECTORS
# ==========================================================

def fft_detector(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape

    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shift))

    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    r_max = np.sqrt(cx ** 2 + cy ** 2) + 1e-10
    r_norm = r / r_max

    low_mask  = r_norm <= 0.33
    mid_mask  = (r_norm > 0.33) & (r_norm <= 0.66)
    high_mask = r_norm > 0.66

    low_energy  = float(np.mean(magnitude[low_mask]))  if low_mask.any()  else 0.0
    mid_energy  = float(np.mean(magnitude[mid_mask]))  if mid_mask.any()  else 0.0
    high_energy = float(np.mean(magnitude[high_mask])) if high_mask.any() else 0.0

    total_energy = float(np.sum(magnitude)) + 1e-10
    high_freq_ratio = float(np.sum(magnitude[high_mask]) / total_energy)

    mag_flat = magnitude.flatten()
    mag_prob = mag_flat / (mag_flat.sum() + 1e-10)
    spectral_entropy = float(-np.sum(mag_prob * np.log(mag_prob + 1e-10)))

    return {
        "fft_low_energy":        low_energy,
        "fft_mid_energy":        mid_energy,
        "fft_high_energy":       high_energy,
        "fft_high_freq_ratio":   high_freq_ratio,
        "fft_entropy":           spectral_entropy,
        "fft_mid_to_high_ratio": float(mid_energy / (high_energy + 1e-10)),
    }

def ela_detector(image):
    img_rgb = image.convert("RGB")

    buffer95 = BytesIO()
    img_rgb.save(buffer95, format="JPEG", quality=95)
    buffer95.seek(0)
    recomp95 = Image.open(buffer95).convert("RGB")
    ela95 = np.array(ImageChops.difference(img_rgb, recomp95)).astype(np.float32)

    buffer75 = BytesIO()
    img_rgb.save(buffer75, format="JPEG", quality=75)
    buffer75.seek(0)
    recomp75 = Image.open(buffer75).convert("RGB")
    ela75 = np.array(ImageChops.difference(img_rgb, recomp75)).astype(np.float32)

    ela95_std = float(np.std(ela95)) + 1e-8

    return {
        "ela_mean_q95":  float(np.mean(ela95)),
        "ela_std_q95":   float(np.std(ela95)),
        "ela_max_q95":   float(np.max(ela95)),
        "ela_mean_q75":  float(np.mean(ela75)),
        "ela_std_q75":   float(np.std(ela75)),
        "ela_skew":      float(np.mean(((ela95 - np.mean(ela95)) / ela95_std) ** 3)),
        "ela_kurtosis":  float(np.mean(((ela95 - np.mean(ela95)) / ela95_std) ** 4)),
    }

def noise_detector(image):
    img = np.array(image).astype(np.float32)

    denoised_gauss = cv2.GaussianBlur(img, (5, 5), 0)
    residual_gauss = img - denoised_gauss

    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    denoised_median = cv2.medianBlur(gray.astype(np.uint8), 5).astype(np.float32)
    residual_median = gray - denoised_median

    laplacian = cv2.Laplacian(gray, cv2.CV_32F)

    n_channels = img.shape[2] if img.ndim == 3 else 1
    channel_stds = [float(np.std(residual_gauss[:, :, c])) for c in range(min(3, n_channels))]
    if len(channel_stds) < 3:
        channel_stds = channel_stds + [0.0] * (3 - len(channel_stds))

    return {
        "noise_std_gauss":         float(np.std(residual_gauss)),
        "noise_mean_gauss":        float(np.mean(np.abs(residual_gauss))),
        "noise_std_median":        float(np.std(residual_median)),
        "noise_laplacian_var":     float(np.var(laplacian)),
        "noise_channel_std_range": float(max(channel_stds) - min(channel_stds)),
        "noise_channel_std_mean":  float(np.mean(channel_stds)),
    }

def metadata_detector(image):
    w, h = image.size
    aspect_ratio = w / max(h, 1)
    total_pixels = w * h

    is_square = (w == h)
    is_common_ai_size = (w, h) in [
        (512, 512), (768, 768), (1024, 1024), (256, 256),
        (512, 768), (768, 512), (1024, 768), (768, 1024),
    ]
    is_power_of_2 = (w > 0 and h > 0 and (w & (w - 1) == 0) and (h & (h - 1) == 0))

    mode = image.mode
    num_channels = len(image.getbands())
    has_alpha = int("A" in mode or mode == "RGBA")

    has_icc_profile = int(image.info.get("icc_profile") is not None)

    return {
        "meta_has_icc_profile":    has_icc_profile,
        "meta_aspect_ratio":       float(aspect_ratio),
        "meta_total_pixels":       float(math.log1p(total_pixels)),
        "meta_is_square":          int(is_square),
        "meta_is_common_ai_size":  int(is_common_ai_size),
        "meta_is_power_of_2":      int(is_power_of_2),
        "meta_has_alpha":          has_alpha,
        "meta_num_channels":       int(num_channels),
    }

def cpu_detectors(image):
    result = {}
    result.update(fft_detector(image))
    result.update(ela_detector(image))
    result.update(noise_detector(image))
    result.update(metadata_detector(image))
    return result

def run_all_detectors_batched(original_img, transformations_dict):
    attack_names = list(transformations_dict.keys())
    
    transformed_images = []
    for attack_name in attack_names:
        transformed = transformations_dict[attack_name](original_img.copy())
        transformed_images.append(transformed)
        
    siglip_results = siglip_batch_detector(transformed_images)
    vit_results = vit_batch_detector(transformed_images)
    
    cpu_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        cpu_results = list(executor.map(cpu_detectors, transformed_images))
        
    final_scores = []
    for i in range(len(attack_names)):
        scores = {}
        scores.update(siglip_results[i])
        scores.update(vit_results[i])
        scores.update(cpu_results[i])
        final_scores.append(scores)
        
    return attack_names, final_scores


# ==========================================================
# 5. GRADIO UI
# ==========================================================

def predict_image(img):
    if not ensemble:
        return "Error: ensemble_v2.joblib not found. Please download it and place it next to this script.", ""
    if not FEATURE_NAMES:
        return "Error: features_v2.json not found. Please download it and place it next to this script.", ""

    try:
        # For a full evaluation, run all 38 transformations
        # To make it faster in the UI, you could change `transformations` here to just `{"none": transformations["none"]}`
        attack_names, batch_scores = run_all_detectors_batched(img.convert("RGB"), transformations)
        
        # Format into a DataFrame exactly how the model expects it
        df = pd.DataFrame(batch_scores)
        
        # Check for missing features just in case
        missing_features = [f for f in FEATURE_NAMES if f not in df.columns]
        if missing_features:
            return f"Error: Missing features during extraction: {missing_features}", ""

        # Reorder columns to perfectly match training
        df = df[FEATURE_NAMES]
        
        # Predict on all 38 transformed versions
        predictions = ensemble.predict(df)
        
        # Simple voting logic: if majority says AI, it's AI
        ai_votes = sum(predictions)
        real_votes = len(predictions) - ai_votes
        
        final_label = "AI Generated" if ai_votes > real_votes else "Real Photo"
        
        details = f"Out of {len(predictions)} transformations evaluated:\n"
        details += f"  - Detected as AI: {ai_votes} times\n"
        details += f"  - Detected as Real: {real_votes} times\n"
        
        return final_label, details

    except Exception as e:
        return f"Error during processing: {str(e)}", ""

# Build Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 AI vs Real Image Forensics Analyzer")
    gr.Markdown("Upload an image to run it through the deep forensics pipeline (evaluates 38 augmented versions).")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Upload an Image")
            analyze_btn = gr.Button("Analyze Image", variant="primary")
        
        with gr.Column():
            output_label = gr.Textbox(label="Final Prediction", lines=1)
            output_details = gr.Textbox(label="Detailed Analysis", lines=5)

    analyze_btn.click(fn=predict_image, inputs=img_input, outputs=[output_label, output_details])

if __name__ == "__main__":
    print("Starting up the Real-Time UI...")
    demo.launch(inbrowser=True)
