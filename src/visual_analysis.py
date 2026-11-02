"""Visual analysis module for FruitSenseAI.

Provides Grad-CAM attention maps and SmolVLM visual description with heuristic cue analysis.
"""

from __future__ import annotations

import gc
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.config import (
    IMAGE_SIZE,
    OUTPUT_WEB_DIR,
    VLM_MAX_NEW_TOKENS,
    VLM_MODEL_ID,
)

# Global caches for VLM to avoid reloading between requests
_VLM_CACHE: dict[str, Any] = {}

def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    predicted_index: int,
    display_image: Image.Image,
) -> tuple[np.ndarray, Image.Image]:
    """Generate Grad-CAM heatmap and blended overlay for MobileNetV3Large.

    Target layer is the final feature layer: model.features[-1].
    """
    model.eval()
    target_layer = model.features[-1]
    captured_activations: dict[str, torch.Tensor] = {}
    captured_gradients: dict[str, torch.Tensor] = {}

    def forward_hook(module, inputs, output):
        captured_activations["value"] = output

    def backward_hook(grad):
        captured_gradients["value"] = grad

    forward_handle = target_layer.register_forward_hook(forward_hook)

    try:
        model.zero_grad(set_to_none=True)
        # Forward pass with gradient tracking
        logits = model(input_tensor)
        activation = captured_activations["value"]
        backward_handle = activation.register_hook(backward_hook)

        target_score = logits[0, predicted_index]
        target_score.backward()
        backward_handle.remove()

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
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam = np.zeros_like(cam)

    finally:
        forward_handle.remove()
        model.zero_grad(set_to_none=True)

    # Create overlay on top of display_image resized to (IMAGE_SIZE, IMAGE_SIZE)
    resized_display = display_image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.asarray(resized_display).astype(np.float32) / 255.0

    # Apply JET colormap
    import matplotlib as mpl
    colormap = mpl.colormaps["jet"]
    heatmap_colored = colormap(cam)[:, :, :3]  # Drop alpha

    # Blend original and heatmap
    alpha = 0.45
    overlay_array = (1.0 - alpha) * img_array + alpha * heatmap_colored
    overlay_array = np.clip(overlay_array * 255.0, 0, 255).astype(np.uint8)
    overlay_image = Image.fromarray(overlay_array)

    return cam, overlay_image

def load_vlm(device: torch.device | None = None) -> tuple[Any, Any]:
    """Load SmolVLM processor and model, cached in memory."""
    device = device or get_device()
    cache_key = f"{VLM_MODEL_ID}_{device.type}"
    if cache_key in _VLM_CACHE:
        return _VLM_CACHE[cache_key]

    vlm_dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        VLM_MODEL_ID,
        dtype=vlm_dtype,
    )
    model = model.to(device)
    model.eval()

    _VLM_CACHE[cache_key] = (processor, model)
    return processor, model

def generate_vlm_description(
    image: Image.Image,
    device: torch.device | None = None,
) -> str:
    """Generate visual appearance description using SmolVLM-500M-Instruct."""
    device = device or get_device()
    try:
        processor, vlm = load_vlm(device=device)

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
            images=[image],
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        with torch.inference_mode():
            generated_ids = vlm.generate(
                **inputs,
                max_new_tokens=VLM_MAX_NEW_TOKENS,
                do_sample=False,
            )

        input_length = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[:, input_length:]
        description = processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
        )[0].strip()

        # Clean VRAM after generation
        del inputs, generated_ids, new_tokens
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return description

    except Exception as exc:
        # Fallback graceful handler if VLM encounters an error
        return f"Visible fruit surface was inspected, but automated description could not be completed ({exc})."

def analyze_vlm_cues(
    vlm_description: str,
    cnn_predicted_class: str,
) -> dict[str, Any]:
    """Conservative keyword heuristic analysis for fruit families and visible deterioration cues."""
    description_lower = vlm_description.lower()

    fruit_keywords = {
        "apples": ["apple", "apples"],
        "banana": ["banana", "bananas"],
        "oranges": ["orange", "oranges", "citrus"],
    }

    cnn_fruit = None
    for fruit in fruit_keywords:
        if fruit in cnn_predicted_class.lower():
            cnn_fruit = fruit
            break

    mentioned_fruits = []
    for fruit, keywords in fruit_keywords.items():
        if any(re.search(r"\b" + re.escape(kw) + r"\b", description_lower) for kw in keywords):
            mentioned_fruits.append(fruit)

    fruit_disagreement = (
        cnn_fruit is not None
        and len(mentioned_fruits) > 0
        and cnn_fruit not in mentioned_fruits
    )

    def cue_status(text: str, pattern: str) -> bool | None:
        negation_pattern = re.compile(
            r"\b(?:no|not|without|free of|does not show|doesn't show|"
            r"do not see|don't see|there is no|there are no)\b"
            r"(?:\s+\w+){0,4}\s+" + pattern,
            re.IGNORECASE,
        )
        if negation_pattern.search(text):
            return False

        positive_pattern = re.compile(r"\b" + pattern + r"\b", re.IGNORECASE)
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

    positive_cues = [name for name, status in cue_results.items() if status is True]
    explicitly_absent_cues = [name for name, status in cue_results.items() if status is False]
    unmentioned_cues = [name for name, status in cue_results.items() if status is None]

    if fruit_disagreement:
        disagreement_status = "FRUIT_LABEL_REVIEW"
    elif positive_cues:
        disagreement_status = "POSSIBLE_VISUAL_DETERIORATION_CUES_MENTIONED"
    elif explicitly_absent_cues:
        disagreement_status = "NO_EXPLICIT_DETERIORATION_CUES_IN_DESCRIPTION"
    else:
        disagreement_status = "INCONCLUSIVE_REQUIRES_REVIEW"

    return {
        "cnn_predicted_fruit_family": cnn_fruit,
        "vlm_mentioned_fruit_families": mentioned_fruits,
        "fruit_label_disagreement_heuristic": fruit_disagreement,
        "visual_cue_status_heuristic": cue_results,
        "possible_deterioration_cues_mentioned": positive_cues,
        "explicitly_absent_cues": explicitly_absent_cues,
        "unmentioned_cues": unmentioned_cues,
        "review_status_heuristic": disagreement_status,
    }

def build_evidence_object(
    cnn_results: dict[str, Any],
    vlm_description: str,
    cue_analysis: dict[str, Any],
    image_path: str = "",
    gradcam_path: str = "",
) -> dict[str, Any]:
    """Assemble the structured evidence dictionary matching the schema expected by Gemini."""
    return {
        "image_path": image_path,
        "cnn_model": "MobileNetV3Large",
        "cnn_predicted_class": cnn_results["predicted_class"],
        "cnn_confidence": cnn_results["confidence"],
        "cnn_class_probabilities": cnn_results["class_probabilities"],
        "gradcam_path": gradcam_path,
        "vlm_model": VLM_MODEL_ID,
        "vlm_description": vlm_description,
        "cnn_predicted_fruit_family": cue_analysis.get("cnn_predicted_fruit_family"),
        "vlm_mentioned_fruit_families": cue_analysis.get("vlm_mentioned_fruit_families", []),
        "fruit_label_disagreement_heuristic": cue_analysis.get("fruit_label_disagreement_heuristic", False),
        "visual_cue_status_heuristic": cue_analysis.get("visual_cue_status_heuristic", {}),
        "possible_deterioration_cues_mentioned": cue_analysis.get("possible_deterioration_cues_mentioned", []),
        "explicitly_absent_cues": cue_analysis.get("explicitly_absent_cues", []),
        "unmentioned_cues": cue_analysis.get("unmentioned_cues", []),
        "review_status_heuristic": cue_analysis.get("review_status_heuristic", "INCONCLUSIVE_REQUIRES_REVIEW"),
        "limitations": [
            "CNN confidence is not a guarantee of correctness.",
            "Grad-CAM highlights influential image regions, not proof of reasoning.",
            "SmolVLM descriptions and keyword analysis are not validated freshness measurements.",
            "Visual appearance cannot establish internal fruit quality or food safety.",
        ],
    }

def save_run_artifacts(
    run_id: str,
    original_image: Image.Image,
    gradcam_overlay: Image.Image,
    evidence: dict[str, Any],
) -> dict[str, str]:
    """Save run artifacts into outputs/web_runs/<run_id>/ without overwriting static files."""
    run_dir = OUTPUT_WEB_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    orig_path = run_dir / "original.png"
    gradcam_path = run_dir / "gradcam_overlay.png"
    json_path = run_dir / "evidence.json"

    original_image.save(orig_path)
    gradcam_overlay.save(gradcam_path)

    evidence["image_path"] = str(orig_path)
    evidence["gradcam_path"] = str(gradcam_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=4, ensure_ascii=False)

    return {
        "original_image_path": str(orig_path),
        "gradcam_overlay_path": str(gradcam_path),
        "evidence_json_path": str(json_path),
    }
