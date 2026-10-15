
from pathlib import Path
from collections import Counter
import argparse
import random
import shutil

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split


# ---------------- CONFIGURATION ----------------

SEED = 42
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"
PLOTS = PROJECT_ROOT / "outputs" / "plots"
REPORTS = PROJECT_ROOT / "outputs" / "reports"

random.seed(SEED)


# ---------------- DATASET DISCOVERY ----------------

def image_files(folder):
    """Return supported images recursively."""
    if not folder.exists():
        return []

    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def class_directories(folder):
    """Return subdirectories containing images."""
    if not folder.is_dir():
        return []

    return sorted(
        p for p in folder.iterdir()
        if p.is_dir() and image_files(p)
    )


def has_train_test_split(folder):
    names = {
        p.name.lower(): p
        for p in folder.iterdir()
        if p.is_dir()
    }
    return (
        "train" in names
        and "test" in names
        and bool(class_directories(names["train"]))
        and bool(class_directories(names["test"]))
    )


def find_dataset_root(raw_dir):
    """
    Search raw_dir and its subdirectories.
    Supports:
      root/train/class_name/
      root/test/class_name/
    or:
      root/class_name/
    """
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {raw_dir}"
        )

    candidates = [raw_dir]
    candidates.extend(
        p for p in raw_dir.rglob("*")
        if p.is_dir()
    )

    # Prefer a directory with an existing train/test split.
    for folder in candidates:
        if has_train_test_split(folder):
            return folder, True

    # Otherwise find a directory containing class folders.
    for folder in candidates:
        if class_directories(folder):
            return folder, False

    raise RuntimeError(
        "Could not locate the dataset. Extract the dataset "
        "inside data/raw/ and check its folder structure."
    )


# ---------------- IMAGE VALIDATION ----------------

def inspect_images(root, split_exists):
    """Check image readability and collect metadata."""
    records = []
    corrupt = []

    if split_exists:
        locations = [
            ("train", root / "train"),
            ("test", root / "test"),
        ]

        validation_dir = next(
            (
                root / name
                for name in ("val", "valid", "validation")
                if (root / name).is_dir()
            ),
            None,
        )

        if validation_dir:
            locations.append(("val", validation_dir))

        for split, folder in locations:
            for class_dir in class_directories(folder):
                for path in image_files(class_dir):
                    try:
                        with Image.open(path) as img:
                            img.verify()

                        with Image.open(path) as img:
                            width, height = img.size

                        records.append({
                            "split": split,
                            "class_name": class_dir.name,
                            "filename": path.name,
                            "width": width,
                            "height": height,
                            "path": str(path),
                        })

                    except Exception as exc:
                        corrupt.append({
                            "path": str(path),
                            "error": str(exc),
                        })

    else:
        for class_dir in class_directories(root):
            for path in image_files(class_dir):
                try:
                    with Image.open(path) as img:
                        img.verify()

                    with Image.open(path) as img:
                        width, height = img.size

                    records.append({
                        "split": "unsplit",
                        "class_name": class_dir.name,
                        "filename": path.name,
                        "width": width,
                        "height": height,
                        "path": str(path),
                    })

                except Exception as exc:
                    corrupt.append({
                        "path": str(path),
                        "error": str(exc),
                    })

    return pd.DataFrame(records), corrupt


# ---------------- PLOTS ----------------

def save_class_distribution(df):
    counts = (
        df.groupby(["split", "class_name"])
        .size()
        .unstack(fill_value=0)
    )

    ax = counts.T.plot(
        kind="bar",
        figsize=(12, 6),
        width=0.8,
    )
    ax.set_title("Fruit Freshness Dataset Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Images")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Split")
    plt.tight_layout()

    output = PLOTS / "class_distribution.png"
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"Saved: {output}")


def save_sample_grid(df, max_classes=6):
    classes = sorted(df["class_name"].unique())[:max_classes]

    if not classes:
        return

    fig, axes = plt.subplots(
        len(classes), 4,
        figsize=(12, 3 * len(classes)),
        squeeze=False,
    )

    for row, class_name in enumerate(classes):
        samples = (
            df[df["class_name"] == class_name]
            .sample(
                n=min(4, len(df[df["class_name"] == class_name])),
                random_state=SEED,
            )
        )

        for col in range(4):
            ax = axes[row][col]
            ax.axis("off")

            if col < len(samples):
                record = samples.iloc[col]

                try:
                    with Image.open(record["path"]) as img:
                        img = ImageOps.exif_transpose(img).convert("RGB")
                        ax.imshow(img)

                    ax.set_title(
                        f"{class_name}\n"
                        f'{record["width"]}x{record["height"]}',
                        fontsize=9,
                    )
                except Exception:
                    ax.set_title("Could not display image")

    plt.tight_layout()
    output = PLOTS / "sample_images.png"
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"Saved: {output}")


# ---------------- SPLIT PREPARATION ----------------

def copy_records(records, split_name):
    """Copy images into processed/split/class folders."""
    for record in records:
        source = Path(record["path"])
        class_name = record["class_name"]

        destination_dir = PROCESSED / split_name / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)

        # Prefix with a unique index to avoid filename collisions.
        destination = destination_dir / (
            f'{record["index"]:06d}_{source.name}'
        )

        shutil.copy2(source, destination)


def prepare_splits(df, root, split_exists):
    """
    Preserve a supplied train/test split.
    If no split exists, create a stratified 70/15/15 split.
    """
    if PROCESSED.exists():
        print(
            f"\nProcessed folder already exists: {PROCESSED}\n"
            "Remove it manually if you want to regenerate splits."
        )
        return

    PROCESSED.mkdir(parents=True, exist_ok=True)

    if split_exists:
        print("\nExisting train/test split detected.")

        train_df = df[df["split"] == "train"].copy()
        test_df = df[df["split"] == "test"].copy()
        val_df = df[df["split"] == "val"].copy()

        # If there is no validation folder, create validation
        # only from the existing training set.
        if val_df.empty:
            train_df, val_df = train_test_split(
                train_df,
                test_size=0.15,
                random_state=SEED,
                stratify=train_df["class_name"],
            )

        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()

    else:
        print(
            "\nNo existing split detected. Creating "
            "70% train / 15% validation / 15% test."
        )

        train_df, temporary_df = train_test_split(
            df,
            test_size=0.30,
            random_state=SEED,
            stratify=df["class_name"],
        )

        val_df, test_df = train_test_split(
            temporary_df,
            test_size=0.50,
            random_state=SEED,
            stratify=temporary_df["class_name"],
        )

        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()

    for split_name, split_df in (
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ):
        split_df = split_df.reset_index(drop=True)
        split_df["index"] = range(len(split_df))
        copy_records(split_df, split_name)

    print("\nPrepared dataset folders:")
    for split_name in ("train", "val", "test"):
        split_dir = PROCESSED / split_name
        counts = {
            p.name: len(image_files(p))
            for p in class_directories(split_dir)
        }

        print(f"\n{split_name.upper()}:")
        for class_name, count in counts.items():
            print(f"  {class_name}: {count}")

    print(f"\nProcessed dataset saved at: {PROCESSED}")


# ---------------- MAIN ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_RAW,
        help="Folder containing the extracted dataset",
    )
    args = parser.parse_args()

    PLOTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FRUITSENSE AI - DATASET ANALYSIS")
    print("=" * 60)

    root, split_exists = find_dataset_root(args.dataset_dir)

    print(f"\nDetected dataset root: {root}")
    print(f"Existing train/test split: {split_exists}")

    df, corrupt = inspect_images(root, split_exists)

    if df.empty:
        raise RuntimeError("No readable images were found.")

    print(f"\nTotal readable images: {len(df)}")
    print(f"Number of classes: {df['class_name'].nunique()}")
    print("\nImages by class:")
    print(df["class_name"].value_counts().sort_index())

    print("\nImage counts by split and class:")
    print(pd.crosstab(df["split"], df["class_name"]))

    print("\nImage dimension summary:")
    print(df[["width", "height"]].describe())

    df.to_csv(REPORTS / "image_inventory.csv", index=False)

    if corrupt:
        pd.DataFrame(corrupt).to_csv(
            REPORTS / "corrupted_images.csv",
            index=False,
        )
        print(f"\nUnreadable images: {len(corrupt)}")
    else:
        print("\nNo corrupted images detected.")

    save_class_distribution(df)
    save_sample_grid(df)

    print("\nPreparing processed splits...")
    prepare_splits(df, root, split_exists)

    print("\nDataset analysis completed.")


if __name__ == "__main__":
    main()
