"""Inference module for FruitSenseAI CNN model.

Handles image validation, preprocessing, model loading, and six-class classification.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

from src.config import (
    CLASS_DISPLAY_NAMES,
    CLASS_NAMES,
    DEFAULT_CHECKPOINT_PATH,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MAX_UPLOAD_SIZE_MB,
    ALLOWED_EXTENSIONS,
)

# Global model cache for non-Streamlit environments
_MODEL_CACHE: dict[str, Any] = {}

def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate_and_load_image(image_input: str | Path | bytes | io.BytesIO | Image.Image) -> Image.Image:
    """Validate and convert an input image to RGB PIL Image.

    Handles size limits, corruption, and EXIF orientation.
    """
    if isinstance(image_input, Image.Image):
        image = image_input
    elif isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {path}")
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: {path.suffix}. "
                f"Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_UPLOAD_SIZE_MB:
            raise ValueError(
                f"Image file size ({file_size_mb:.1f} MB) exceeds maximum limit "
                f"of {MAX_UPLOAD_SIZE_MB} MB."
            )
        try:
            image = Image.open(path)
            image.load()
        except Exception as exc:
            raise ValueError(f"Corrupted or invalid image file: {exc}") from exc
    elif isinstance(image_input, (bytes, io.BytesIO)):
        data = image_input.getvalue() if isinstance(image_input, io.BytesIO) else image_input
        size_mb = len(data) / (1024 * 1024)
        if size_mb > MAX_UPLOAD_SIZE_MB:
            raise ValueError(
                f"Uploaded image ({size_mb:.1f} MB) exceeds limit of {MAX_UPLOAD_SIZE_MB} MB."
            )
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except Exception as exc:
            raise ValueError(f"Unable to decode uploaded image: {exc}") from exc
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Correct EXIF rotation if present
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    # Convert to RGB cleanly (removes alpha or palette channels)
    return image.convert("RGB")

def load_cnn_model(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    device: torch.device | None = None,
) -> tuple[nn.Module, list[str]]:
    """Load the trained MobileNetV3Large checkpoint.

    Verifies checkpoint structure and loads weights.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained CNN checkpoint not found at: {path}")

    device = device or get_device()
    cache_key = f"{path.resolve()}_{device.type}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    class_names = checkpoint.get("class_names", CLASS_NAMES)
    num_classes = len(class_names)

    model = models.mobilenet_v3_large(weights=None)
    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features,
        num_classes,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    _MODEL_CACHE[cache_key] = (model, class_names)
    return model, class_names

def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Return the exact CNN evaluation transforms and matching display crop transform."""
    cnn_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    crop_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
    ])
    return cnn_transform, crop_transform

def predict_fruit(
    image: Image.Image,
    model: nn.Module | None = None,
    class_names: list[str] | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run CNN classification on an RGB image.

    Returns predicted class, label, condition, confidence, and probability distribution.
    """
    device = device or get_device()
    if model is None or class_names is None:
        model, class_names = load_cnn_model(device=device)

    cnn_transform, crop_transform = get_transforms()
    input_tensor = cnn_transform(image).unsqueeze(0).to(device)
    cropped_image = crop_transform(image)

    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_idx = int(probabilities.argmax().item())
        predicted_class = class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx].item())

    prob_dict = {
        name: float(probabilities[i].item())
        for i, name in enumerate(class_names)
    }

    # Determine fruit condition and family
    condition = "Fresh" if "fresh" in predicted_class.lower() else "Rotten"
    fruit_family = "unknown"
    for family in ["apples", "banana", "oranges"]:
        if family in predicted_class.lower():
            fruit_family = family
            break

    return {
        "predicted_class": predicted_class,
        "predicted_label": CLASS_DISPLAY_NAMES.get(predicted_class, predicted_class),
        "predicted_condition": condition,
        "fruit_family": fruit_family,
        "confidence": confidence,
        "class_probabilities": prob_dict,
        "predicted_index": predicted_idx,
        "input_tensor": input_tensor,
        "cropped_image": cropped_image,
    }
