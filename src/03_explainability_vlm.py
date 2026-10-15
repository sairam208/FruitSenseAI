from pathlib import Path
import gc
import json
import random
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"G:\project1\Prototype1\FruitSenseAI")
DATASET_DIR = PROJECT_DIR / "data" / "processed" / "dataset"
TEST_DIR = DATASET_DIR / "test"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "explainability_vlm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "outputs"
    / "six_model_comparison"
    / "checkpoints"
    / "MobileNetV3Large_best.pth"
)

# Set to a specific image path to analyze it.
# Keep None to randomly select one test image.
IMAGE_PATH = r"G:\project1\Prototype1\FruitSenseAI\data\processed\dataset\test\freshapples\rotated_by_75_Screen Shot 2018-06-08 at 5.14.44 PM.png"

VLM_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
IMAGE_SIZE = 224
MAX_NEW_TOKENS = 120
SEED = 42

CAM_PATH = OUTPUT_DIR / "gradcam_overlay.png"
COMPARISON_PATH = OUTPUT_DIR / "cnn_vlm_comparison.png"
JSON_PATH = OUTPUT_DIR / "analysis_results.json"
CSV_PATH = OUTPUT_DIR / "analysis_results.csv"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("FRUITSENSE AI - GRAD-CAM + SMOLVLM")
print("=" * 70)
print("Device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 2. VALIDATE FILES
# ============================================================

if not CHECKPOINT_PATH.is_file():
    raise FileNotFoundError(
        f"CNN checkpoint not found:\n{CHECKPOINT_PATH}"
    )

if IMAGE_PATH is not None:
    IMAGE_PATH = Path(IMAGE_PATH)
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
else:
    if not TEST_DIR.is_dir():
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")


# ============================================================
# 3. LOAD THE TRAINED CNN
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
)

class_names = checkpoint["class_names"]
num_classes = len(class_names)

cnn = models.mobilenet_v3_large(weights=None)
cnn.classifier[-1] = nn.Linear(
    cnn.classifier[-1].in_features,
    num_classes,
)
cnn.load_state_dict(checkpoint["model_state_dict"])
cnn = cnn.to(DEVICE)
cnn.eval()

print("\nCNN loaded:", checkpoint.get("model_name", "MobileNetV3Large"))
print("Classes:", class_names)


# ============================================================
# 4. SELECT IMAGE AND PREPROCESS
# ============================================================

if IMAGE_PATH is None:
    supported_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    candidate_images = [
        path for path in TEST_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in supported_extensions
    ]
    if not candidate_images:
        raise RuntimeError(f"No supported images found in {TEST_DIR}")
    candidate_images.sort()
    IMAGE_PATH = random.choice(candidate_images)

print("\nAnalyzing image:", IMAGE_PATH)

with Image.open(IMAGE_PATH) as image_file:
    original_image = image_file.convert("RGB")

display_image = original_image.copy()

cnn_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])

input_tensor = cnn_transform(original_image).unsqueeze(0).to(DEVICE)

# Keep display crop aligned with the CNN input.
cropped_image = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
])(display_image)
image_array = np.asarray(cropped_image).astype(np.float32) / 255.0


# ============================================================
# 5. CNN PREDICTION
# ============================================================

with torch.inference_mode():
    logits = cnn(input_tensor)
    probabilities = torch.softmax(logits, dim=1)[0]
    predicted_index = int(probabilities.argmax().item())
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index].item())

probability_results = {
    class_names[i]: float(probabilities[i].item())
    for i in range(num_classes)
}

print("\n" + "=" * 70)
print("CNN PREDICTION")
print("=" * 70)
print("Predicted class:", predicted_class)
print(f"Confidence: {confidence:.4%}")
print("\nAll class probabilities:")
for name, probability in sorted(
    probability_results.items(),
    key=lambda item: item[1],
    reverse=True,
):
    print(f"{name:20s}: {probability:.4%}")


# ============================================================
# 6. GRAD-CAM
# ============================================================

# MobileNetV3's final convolutional feature block.
target_layer = cnn.features[-1]
captured_activations = {}
captured_gradients = {}

def save_activation(module, inputs, output):
    captured_activations["value"] = output

def save_gradient(gradient):
    captured_gradients["value"] = gradient

forward_handle = target_layer.register_forward_hook(save_activation)

try:
    cnn.zero_grad(set_to_none=True)

    # Must NOT run this section inside inference_mode/no_grad.
    gradcam_logits = cnn(input_tensor)
    activation = captured_activations["value"]
    activation.register_hook(save_gradient)

    target_score = gradcam_logits[0, predicted_index]
    target_score.backward()

    gradient = captured_gradients["value"]
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam_tensor = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    cam_tensor = F.interpolate(
        cam_tensor,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    cam = cam_tensor[0, 0].detach().cpu().numpy()
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)

finally:
    forward_handle.remove()
    cnn.zero_grad(set_to_none=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image_array)
axes[0].set_title("CNN Input")
axes[0].axis("off")

axes[1].imshow(cam, cmap="jet")
axes[1].set_title("Grad-CAM Heatmap")
axes[1].axis("off")

axes[2].imshow(image_array)
axes[2].imshow(cam, cmap="jet", alpha=0.45)
axes[2].set_title(f"Grad-CAM Overlay\n{predicted_class}")
axes[2].axis("off")

plt.tight_layout()
plt.savefig(CAM_PATH, dpi=200, bbox_inches="tight")
plt.close(fig)
print("\nGrad-CAM saved:", CAM_PATH)


# ============================================================
# 7. RELEASE CNN GPU MEMORY BEFORE LOADING VLM
# ============================================================

# Keep the computed results and CAM, but move the CNN off GPU.
cnn = cnn.to("cpu")
del input_tensor, logits, gradcam_logits, activation, gradient
del cam_tensor, probabilities, target_score
gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# 8. LOAD SMOLVLM
# ============================================================

print("\n" + "=" * 70)
print("LOADING SMOLVLM")
print("=" * 70)

try:
    from transformers import AutoProcessor, AutoModelForImageTextToText
except ImportError as exc:
    raise ImportError(
        "Your Transformers version does not provide "
        "AutoModelForImageTextToText. Try:\n"
        "uv add --upgrade transformers"
    ) from exc

vlm_device = DEVICE
vlm_dtype = torch.float16 if vlm_device.type == "cuda" else torch.float32

processor = AutoProcessor.from_pretrained(VLM_ID)
vlm = AutoModelForImageTextToText.from_pretrained(
    VLM_ID,
    torch_dtype=vlm_dtype,
)
vlm = vlm.to(vlm_device)
vlm.eval()

print("VLM loaded:", VLM_ID)


# ============================================================
# 9. ASK VLM FOR A VISUAL DESCRIPTION
# ============================================================

prompt_text = (
    "Inspect this fruit image. Describe the visible fruit type, color, "
    "surface texture, visible spots, bruises, mold-like areas, "
    "discoloration, and other observable visual features. "
    "Only describe what is visible. Explicitly distinguish features "
    "you see from features you do not see. Do not infer hidden internal "
    "quality or claim certainty about freshness from appearance alone. "
    "Keep the answer concise."
)

messages = [{
    "role": "user",
    "content": [
        {"type": "image"},
        {"type": "text", "text": prompt_text},
    ],
}]

prompt = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
)

inputs = processor(
    text=prompt,
    images=[display_image],
    return_tensors="pt",
)
inputs = inputs.to(vlm_device)

print("\nGenerating visual description...")
with torch.inference_mode():
    generated_ids = vlm.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )

# Decode only generated tokens, excluding the prompt.
input_length = inputs["input_ids"].shape[1]
new_tokens = generated_ids[:, input_length:]
vlm_description = processor.batch_decode(
    new_tokens,
    skip_special_tokens=True,
)[0].strip()

print("\n" + "=" * 70)
print("SMOLVLM DESCRIPTION")
print("=" * 70)
print(vlm_description)


# ============================================================
# 10. CONSERVATIVE CNN-VLM TEXT ANALYSIS
# ============================================================

# Important:
# SmolVLM produces free text, not a calibrated 6-class prediction.
# These rules are heuristics and must not be treated as ground truth.

description_lower = vlm_description.lower()

fruit_keywords = {
    "apples": ["apple", "apples"],
    "banana": ["banana", "bananas"],
    "oranges": ["orange", "oranges", "citrus"],
}

cnn_fruit = None
for fruit in fruit_keywords:
    if fruit in predicted_class.lower():
        cnn_fruit = fruit
        break

mentioned_fruits = []
for fruit, keywords in fruit_keywords.items():
    if any(re.search(r"\b" + re.escape(keyword) + r"\b", description_lower)
           for keyword in keywords):
        mentioned_fruits.append(fruit)

# Only flag disagreement if the VLM explicitly names a fruit family
# and the family differs from the CNN's family.
fruit_disagreement = (
    cnn_fruit is not None
    and len(mentioned_fruits) > 0
    and cnn_fruit not in mentioned_fruits
)

def cue_status(text, cue_pattern):
    """
    Heuristic cue detector:
      True  = cue mentioned without recognized negation
      False = cue appears explicitly negated
      None  = cue not mentioned

    Negation handling is intentionally conservative but is not a
    substitute for a validated natural-language classifier.
    """
    negation_pattern = re.compile(
        r"\b(?:no|not|without|free of|does not show|doesn't show|"
        r"do not see|don't see|there is no|there are no)\b"
        r"(?:\s+\w+){0,4}\s+" + cue_pattern,
        re.IGNORECASE,
    )
    if negation_pattern.search(text):
        return False

    positive_pattern = re.compile(
        r"\b" + cue_pattern + r"\b",
        re.IGNORECASE,
    )
    if positive_pattern.search(text):
        return True

    return None

cue_patterns = {
    "mold": r"mold(?:-like)?|mould",
    "bruising": r"bruis(?:e|es|ed|ing)",
    "discoloration": r"discoloration|discolouration|discolored|discoloured",
    "rot_or_decay": r"rot|rotten|decay",
    "spots_or_blemishes": r"spots?|blemishes",
    "wrinkling": r"wrinkl(?:e|es|ed|ing)",
}

cue_results = {
    name: cue_status(description_lower, pattern)
    for name, pattern in cue_patterns.items()
}

positive_cues = [
    name for name, status in cue_results.items() if status is True
]
explicitly_absent_cues = [
    name for name, status in cue_results.items() if status is False
]
unmentioned_cues = [
    name for name, status in cue_results.items() if status is None
]

if fruit_disagreement:
    disagreement_status = "FRUIT_LABEL_REVIEW"
elif positive_cues:
    disagreement_status = "POSSIBLE_VISUAL_DETERIORATION_CUES_MENTIONED"
elif explicitly_absent_cues:
    disagreement_status = "NO_EXPLICIT_DETERIORATION_CUES_IN_DESCRIPTION"
else:
    disagreement_status = "INCONCLUSIVE_REQUIRES_REVIEW"

print("\n" + "=" * 70)
print("CNN-VLM TEXT ANALYSIS (HEURISTIC)")
print("=" * 70)
print("CNN fruit family:", cnn_fruit)
print("Fruit families explicitly mentioned by VLM:", mentioned_fruits)
print("Fruit label disagreement:", fruit_disagreement)
print("Cue analysis:", cue_results)
print("Possible deterioration cues mentioned:", positive_cues)
print("Explicitly absent cues:", explicitly_absent_cues)
print("Unmentioned cues:", unmentioned_cues)
print("Review status:", disagreement_status)
print(
    "\nWARNING: This text analysis is heuristic. It does not establish "
    "fruit freshness or food safety."
)


# ============================================================
# 11. COMBINED VISUAL REPORT
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].imshow(display_image)
axes[0].set_title(
    f"CNN: {predicted_class}\nConfidence: {confidence:.2%}"
)
axes[0].axis("off")

axes[1].imshow(image_array)
axes[1].imshow(cam, cmap="jet", alpha=0.45)
axes[1].set_title("Grad-CAM: Regions influencing CNN")
axes[1].axis("off")

plt.tight_layout()
plt.savefig(COMPARISON_PATH, dpi=200, bbox_inches="tight")
plt.close(fig)


# ============================================================
# 12. SAVE JSON AND CSV REPORTS
# ============================================================

results = {
    "image_path": str(IMAGE_PATH),
    "cnn_model": "MobileNetV3Large",
    "cnn_predicted_class": predicted_class,
    "cnn_confidence": confidence,
    "cnn_class_probabilities": probability_results,
    "gradcam_path": str(CAM_PATH),
    "combined_visualization_path": str(COMPARISON_PATH),
    "vlm_model": VLM_ID,
    "vlm_description": vlm_description,
    "cnn_predicted_fruit_family": cnn_fruit,
    "vlm_mentioned_fruit_families": mentioned_fruits,
    "fruit_label_disagreement_heuristic": fruit_disagreement,
    "visual_cue_status_heuristic": cue_results,
    "possible_deterioration_cues_mentioned": positive_cues,
    "explicitly_absent_cues": explicitly_absent_cues,
    "unmentioned_cues": unmentioned_cues,
    "review_status_heuristic": disagreement_status,
    "limitations": [
        "CNN confidence is not a guarantee of correctness.",
        "Grad-CAM highlights influential image regions, not proof of reasoning.",
        "SmolVLM descriptions and keyword analysis are not validated freshness measurements.",
        "Visual appearance cannot establish internal fruit quality or food safety.",
    ],
}

with open(JSON_PATH, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4, ensure_ascii=False)

pd.DataFrame([{
    "image_path": str(IMAGE_PATH),
    "cnn_predicted_class": predicted_class,
    "cnn_confidence": confidence,
    "vlm_description": vlm_description,
    "cnn_predicted_fruit_family": cnn_fruit,
    "vlm_mentioned_fruit_families": ", ".join(mentioned_fruits),
    "fruit_label_disagreement_heuristic": fruit_disagreement,
    "visual_cue_status_heuristic": json.dumps(cue_results),
    "review_status_heuristic": disagreement_status,
}]).to_csv(CSV_PATH, index=False)

print("\n" + "=" * 70)
print("EXPLAINABILITY + VLM ANALYSIS COMPLETED")
print("=" * 70)
print("Grad-CAM:", CAM_PATH)
print("Combined visualization:", COMPARISON_PATH)
print("JSON report:", JSON_PATH)
print("CSV report:", CSV_PATH)

# Release model and inference memory.
del inputs, generated_ids, vlm, processor
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
