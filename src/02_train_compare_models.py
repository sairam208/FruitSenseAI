from pathlib import Path
import gc
import json
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from tqdm.auto import tqdm


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"G:\project1\Prototype1\FruitSenseAI")
DATASET_DIR = PROJECT_DIR / "data" / "processed" / "dataset"

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "six_model_comparison"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PLOT_DIR = OUTPUT_DIR / "plots"
REPORT_DIR = OUTPUT_DIR / "reports"

for folder in [CHECKPOINT_DIR, PLOT_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 6
VALIDATION_FRACTION = 0.15
LEARNING_RATE = 0.0001

MODEL_NAMES = [
    "MobileNetV2",
    "MobileNetV3Large",
    "ResNet50",
    "DenseNet121",
    "EfficientNetB0",
    "ConvNeXtTiny",
]

# RESUME SETTINGS
# MobileNetV2 has already finished.
# Set to 1 to resume from MobileNetV2 if needed.
START_MODEL_INDEX = 1

# If a completed model has a valid result and checkpoint,
# skip it automatically.
SKIP_COMPLETED_MODELS = True

# Save the comparison CSV after every completed model.
COMPARISON_CSV = REPORT_DIR / "six_model_comparison.csv"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

print("=" * 75)
print("FRUITSENSE AI - RESUMABLE SIX CNN EXPERIMENT")
print("=" * 75)
print("Dataset:", DATASET_DIR)
print("Device:", DEVICE)
print("Mixed precision:", USE_AMP)
print("Starting model:", MODEL_NAMES[START_MODEL_INDEX])

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2,
        ),
        "GB",
    )

if not TRAIN_DIR.is_dir():
    raise FileNotFoundError(f"Training folder not found: {TRAIN_DIR}")

if not TEST_DIR.is_dir():
    raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")


# ============================================================
# 2. IMAGE TRANSFORMS
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.85, 1.0),
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(
        brightness=0.12,
        contrast=0.12,
        saturation=0.08,
    ),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ============================================================
# 3. LOAD DATASET
# ============================================================

train_aug_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform,
)

train_eval_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=eval_transform,
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=eval_transform,
)

class_names = train_aug_dataset.classes
num_classes = len(class_names)

if train_aug_dataset.class_to_idx != test_dataset.class_to_idx:
    raise ValueError(
        "Train and test class names/order do not match."
    )

if num_classes != 6:
    raise ValueError(
        f"Expected six classes, found {num_classes}: {class_names}"
    )

if len(train_aug_dataset) == 0 or len(test_dataset) == 0:
    raise ValueError("Training or test dataset is empty.")

all_labels = np.array([
    label for _, label in train_aug_dataset.samples
])

all_indices = np.arange(len(all_labels))

train_indices, val_indices = train_test_split(
    all_indices,
    test_size=VALIDATION_FRACTION,
    random_state=SEED,
    stratify=all_labels,
)

train_dataset = Subset(
    train_aug_dataset,
    train_indices.tolist(),
)

val_dataset = Subset(
    train_eval_dataset,
    val_indices.tolist(),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=USE_AMP,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=USE_AMP,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=USE_AMP,
)

print("\nClasses:", class_names)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images:", len(test_dataset))

with open(
    REPORT_DIR / "class_names.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(class_names, f, indent=4)


# ============================================================
# 4. BUILD MODELS
# ============================================================

def build_model(name, pretrained=True):
    if name == "MobileNetV2":
        weights = (
            models.MobileNet_V2_Weights.DEFAULT
            if pretrained else None
        )
        model = models.mobilenet_v2(weights=weights)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            num_classes,
        )

    elif name == "MobileNetV3Large":
        weights = (
            models.MobileNet_V3_Large_Weights.DEFAULT
            if pretrained else None
        )
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            num_classes,
        )

    elif name == "ResNet50":
        weights = (
            models.ResNet50_Weights.DEFAULT
            if pretrained else None
        )
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes,
        )

    elif name == "DenseNet121":
        weights = (
            models.DenseNet121_Weights.DEFAULT
            if pretrained else None
        )
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(
            model.classifier.in_features,
            num_classes,
        )

    elif name == "EfficientNetB0":
        weights = (
            models.EfficientNet_B0_Weights.DEFAULT
            if pretrained else None
        )
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            num_classes,
        )

    elif name == "ConvNeXtTiny":
        weights = (
            models.ConvNeXt_Tiny_Weights.DEFAULT
            if pretrained else None
        )
        model = models.convnext_tiny(weights=weights)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            num_classes,
        )

    else:
        raise ValueError(f"Unknown model: {name}")

    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# ============================================================
# 5. RECOVER COMPLETED MODEL RESULTS
# ============================================================

def recover_model_result(name):
    """
    Recover a model's result from its existing checkpoint
    and training history. Does not train the model again.
    """

    checkpoint_path = CHECKPOINT_DIR / f"{name}_best.pth"
    history_path = REPORT_DIR / f"{name}_history.csv"

    if not checkpoint_path.is_file():
        return None

    if not history_path.is_file():
        print(
            f"Cannot recover {name}: history CSV is missing."
        )
        return None

    history_df = pd.read_csv(history_path)

    required_columns = {
        "epoch",
        "val_accuracy",
        "val_macro_f1",
        "val_loss",
        "epoch_time_seconds",
    }

    if not required_columns.issubset(history_df.columns):
        print(f"Cannot recover {name}: history is incomplete.")
        return None

    if history_df.empty:
        return None

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint.get("model_name") != name:
        print(f"Checkpoint name mismatch for {name}.")
        return None

    best_row = history_df.loc[
        history_df["val_macro_f1"].idxmax()
    ]

    # Count parameters without downloading pretrained weights.
    model = build_model(name, pretrained=False)
    params = count_parameters(model)
    del model

    result = {
        "model": name,
        "parameters": params,
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_val_accuracy": float(best_row["val_accuracy"]),
        "best_val_macro_f1": float(
            checkpoint["best_val_macro_f1"]
        ),
        "best_val_loss": float(best_row["val_loss"]),
        "training_time_minutes": round(
            history_df["epoch_time_seconds"].sum() / 60,
            2,
        ),
        "checkpoint_size_mb": round(
            checkpoint_path.stat().st_size / (1024 * 1024),
            2,
        ),
    }

    print(f"Recovered saved result for {name}.")
    return result


# ============================================================
# 6. TRAINING / VALIDATION EPOCH
# ============================================================

def run_epoch(model, loader, criterion, optimizer=None, desc=""):
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0
    all_true = []
    all_pred = []

    scaler = getattr(run_epoch, "scaler", None)

    progress = tqdm(
        loader,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
    )

    for images, targets in progress:
        images = images.to(
            DEVICE,
            non_blocking=USE_AMP,
        )
        targets = targets.to(
            DEVICE,
            non_blocking=USE_AMP,
        )

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=USE_AMP,
            ):
                logits = model(images)
                loss = criterion(logits, targets)

            if is_training:
                if USE_AMP and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_count = targets.size(0)

        total_loss += loss.item() * batch_count
        total_samples += batch_count

        predictions = logits.argmax(dim=1)

        all_true.extend(targets.detach().cpu().tolist())
        all_pred.extend(predictions.detach().cpu().tolist())

        running_accuracy = accuracy_score(
            all_true,
            all_pred,
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{running_accuracy:.4f}",
        )

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": accuracy_score(all_true, all_pred),
        "macro_f1": f1_score(
            all_true,
            all_pred,
            average="macro",
            zero_division=0,
        ),
        "true": all_true,
        "pred": all_pred,
    }


# ============================================================
# 7. TRAIN ONE MODEL
# ============================================================

def train_one_model(name):
    print("\n" + "#" * 75)
    print(f"STARTING MODEL: {name}")
    print("#" * 75)

    model = build_model(name, pretrained=True).to(DEVICE)

    params = count_parameters(model)
    print(f"Parameters: {params:,}")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.3,
        patience=2,
    )

    run_epoch.scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP,
    )

    checkpoint_path = CHECKPOINT_DIR / f"{name}_best.pth"

    history = []
    best_val_f1 = -1.0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            desc=f"{name} | Epoch {epoch}/{EPOCHS} | Train",
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            desc=f"{name} | Epoch {epoch}/{EPOCHS} | Validation",
        )

        scheduler.step(val_metrics["macro_f1"])

        elapsed = time.time() - epoch_start

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_time_seconds": elapsed,
        })

        print(
            f"\n{name} | Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss={train_metrics['loss']:.4f} | "
            f"Train Acc={train_metrics['accuracy']:.4f} | "
            f"Val Loss={val_metrics['loss']:.4f} | "
            f"Val Acc={val_metrics['accuracy']:.4f} | "
            f"Val Macro-F1={val_metrics['macro_f1']:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch

            cpu_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            torch.save(
                {
                    "model_name": name,
                    "model_state_dict": cpu_state,
                    "class_names": class_names,
                    "image_size": IMAGE_SIZE,
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_val_f1,
                },
                checkpoint_path,
            )

            print("Saved best validation checkpoint.")

        # Save history after EVERY epoch.
        # This protects completed epoch metrics if interrupted.
        pd.DataFrame(history).to_csv(
            REPORT_DIR / f"{name}_history.csv",
            index=False,
        )

    total_training_time = time.time() - start_time

    history_df = pd.DataFrame(history)

    # Accuracy plot
    plt.figure(figsize=(8, 5))

    plt.plot(
        history_df["epoch"],
        history_df["train_accuracy"],
        marker="o",
        label="Training Accuracy",
    )
    plt.plot(
        history_df["epoch"],
        history_df["val_accuracy"],
        marker="s",
        label="Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{name} - Accuracy vs Epoch")
    plt.xticks(range(1, EPOCHS + 1))
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / f"{name}_accuracy_vs_epoch.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    # Loss plot
    plt.figure(figsize=(8, 5))

    plt.plot(
        history_df["epoch"],
        history_df["train_loss"],
        marker="o",
        label="Training Loss",
    )
    plt.plot(
        history_df["epoch"],
        history_df["val_loss"],
        marker="s",
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{name} - Loss vs Epoch")
    plt.xticks(range(1, EPOCHS + 1))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / f"{name}_loss_vs_epoch.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    best_row = history_df.loc[
        history_df["val_macro_f1"].idxmax()
    ]

    result = {
        "model": name,
        "parameters": params,
        "best_epoch": best_epoch,
        "best_val_accuracy": float(best_row["val_accuracy"]),
        "best_val_macro_f1": float(best_val_f1),
        "best_val_loss": float(best_row["val_loss"]),
        "training_time_minutes": round(
            total_training_time / 60,
            2,
        ),
        "checkpoint_size_mb": round(
            checkpoint_path.stat().st_size / (1024 * 1024),
            2,
        ),
    }

    print(f"\nFinished {name}")
    print(f"Best validation macro-F1: {best_val_f1:.4f}")
    print(
        f"Training time: {total_training_time / 60:.2f} minutes"
    )

    del model, optimizer, scheduler
    run_epoch.scaler = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================
# 8. RESUME / SKIP COMPLETED MODELS
# ============================================================

results_by_name = {}

# First, recover the already completed MobileNetV2.
# Its existing checkpoint and history are reused.
for model_name in MODEL_NAMES[:START_MODEL_INDEX]:
    recovered = recover_model_result(model_name)

    if recovered is not None:
        results_by_name[model_name] = recovered
    else:
        raise RuntimeError(
            f"Cannot recover {model_name} from its existing files.\n"
            "Check its checkpoint and history CSV before continuing."
        )

# Recover other models too, if they were completed in an earlier run.
if SKIP_COMPLETED_MODELS:
    for model_name in MODEL_NAMES[START_MODEL_INDEX:]:
        recovered = recover_model_result(model_name)

        if recovered is not None:
            results_by_name[model_name] = recovered

# Train only the models that have not been completed.
for model_index in range(START_MODEL_INDEX, len(MODEL_NAMES)):
    model_name = MODEL_NAMES[model_index]

    if model_name in results_by_name:
        print(
            f"\nSkipping {model_name}: "
            "completed result and checkpoint already exist."
        )
        continue

    print(
        f"\nOVERALL PROGRESS: MODEL "
        f"{model_index + 1}/{len(MODEL_NAMES)}"
    )

    try:
        result = train_one_model(model_name)
        results_by_name[model_name] = result

        # Save partial comparison immediately after each model.
        partial_results = [
            results_by_name[name]
            for name in MODEL_NAMES
            if name in results_by_name
        ]

        pd.DataFrame(partial_results).to_csv(
            COMPARISON_CSV,
            index=False,
        )

        print(f"Progress saved to: {COMPARISON_CSV}")

    except torch.cuda.OutOfMemoryError:
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(
            f"\nCUDA OUT OF MEMORY for {model_name}.\n"
            "Reduce BATCH_SIZE from 8 to 4 and rerun.\n"
            "Completed models with valid saved checkpoints "
            "will be skipped."
        )
        raise


# ============================================================
# 9. MODEL COMPARISON AND RANKING
# ============================================================

missing_models = [
    name for name in MODEL_NAMES
    if name not in results_by_name
]

if missing_models:
    raise RuntimeError(
        f"Missing model results: {missing_models}"
    )

comparison = pd.DataFrame([
    results_by_name[name]
    for name in MODEL_NAMES
])

comparison = comparison.sort_values(
    ["best_val_macro_f1", "best_val_accuracy"],
    ascending=False,
).reset_index(drop=True)

comparison.insert(
    0,
    "rank",
    range(1, len(comparison) + 1),
)

comparison.to_csv(
    COMPARISON_CSV,
    index=False,
)

print("\n" + "=" * 75)
print("SIX MODEL COMPARISON")
print("=" * 75)
print(comparison.to_string(index=False))


# Combined validation accuracy graph
plt.figure(figsize=(11, 6))

plt.bar(
    comparison["model"],
    comparison["best_val_accuracy"],
)

plt.xlabel("CNN Model")
plt.ylabel("Best Validation Accuracy")
plt.title("FruitSense AI - Six Model Accuracy Comparison")
plt.ylim(0, 1.02)
plt.xticks(rotation=25, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "six_model_accuracy_comparison.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()


# Combined validation macro-F1 graph
plt.figure(figsize=(11, 6))

plt.bar(
    comparison["model"],
    comparison["best_val_macro_f1"],
)

plt.xlabel("CNN Model")
plt.ylabel("Best Validation Macro-F1")
plt.title("FruitSense AI - Six Model Macro-F1 Comparison")
plt.ylim(0, 1.02)
plt.xticks(rotation=25, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "six_model_macro_f1_comparison.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()


# Top three models
top_three = comparison.head(3)

top_three.to_csv(
    REPORT_DIR / "top_three_models.csv",
    index=False,
)

print("\n" + "=" * 75)
print("TOP THREE MODELS - VALIDATION")
print("=" * 75)

print(
    top_three[
        [
            "rank",
            "model",
            "best_val_accuracy",
            "best_val_macro_f1",
            "training_time_minutes",
            "checkpoint_size_mb",
        ]
    ].to_string(index=False)
)


# ============================================================
# 10. FINAL TEST EVALUATION OF THE WINNER
# ============================================================

best_name = comparison.iloc[0]["model"]

best_checkpoint_path = (
    CHECKPOINT_DIR / f"{best_name}_best.pth"
)

print("\nSelected winner:", best_name)
print("Loading best validation checkpoint...")

checkpoint = torch.load(
    best_checkpoint_path,
    map_location="cpu",
    weights_only=True,
)

best_model = build_model(best_name, pretrained=False)

best_model.load_state_dict(
    checkpoint["model_state_dict"]
)

best_model = best_model.to(DEVICE)
best_model.eval()

run_epoch.scaler = None

test_start = time.time()

test_metrics = run_epoch(
    best_model,
    test_loader,
    nn.CrossEntropyLoss(),
    desc=f"{best_name} | Final Test",
)

if DEVICE.type == "cuda":
    torch.cuda.synchronize()

test_time = time.time() - test_start

y_true = test_metrics["true"]
y_pred = test_metrics["pred"]

test_accuracy = accuracy_score(y_true, y_pred)

test_precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

test_recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

test_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

print("\n" + "=" * 75)
print("WINNING MODEL - FINAL TEST RESULTS")
print("=" * 75)
print("Model:", best_name)
print(f"Accuracy:        {test_accuracy:.4f}")
print(f"Macro precision: {test_precision:.4f}")
print(f"Macro recall:    {test_recall:.4f}")
print(f"Macro F1:        {test_f1:.4f}")
print("Test images:", len(test_dataset))
print(f"Test time:       {test_time:.2f} seconds")

print(
    "Average evaluation time/image: "
    f"{1000 * test_time / len(test_dataset):.2f} ms"
)

print("\nPer-class test classification report:")

print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0,
    )
)

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(num_classes)),
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)

pd.DataFrame(report).transpose().to_csv(
    REPORT_DIR / "winning_model_classification_report.csv"
)


# Confusion matrix
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(num_classes)),
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)

fig, ax = plt.subplots(figsize=(9, 8))

disp.plot(
    ax=ax,
    xticks_rotation=45,
    values_format="d",
    cmap="Blues",
)

ax.set_title(f"{best_name} - Test Confusion Matrix")
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "winning_model_confusion_matrix.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)


# Final JSON
final_results = {
    "winning_model": best_name,
    "selection_metric": "validation_macro_f1",
    "best_validation_macro_f1": float(
        comparison.iloc[0]["best_val_macro_f1"]
    ),
    "test_accuracy": float(test_accuracy),
    "test_macro_precision": float(test_precision),
    "test_macro_recall": float(test_recall),
    "test_macro_f1": float(test_f1),
    "test_images": len(test_dataset),
    "test_evaluation_time_seconds": round(test_time, 3),
    "class_names": class_names,
}

with open(
    REPORT_DIR / "final_test_results.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(final_results, f, indent=4)


# ============================================================
# 11. COMPLETION
# ============================================================

print("\n" + "=" * 75)
print("EXPERIMENT COMPLETED")
print("=" * 75)
print("Comparison:", COMPARISON_CSV)
print("Top three:", REPORT_DIR / "top_three_models.csv")
print("Graphs:", PLOT_DIR)
print("Reports:", REPORT_DIR)
print("Best checkpoint:", best_checkpoint_path)

del best_model
gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()