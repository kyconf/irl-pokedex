"""
cleanup.py

Post-scrape cleaning pass for data/raw/<pokemon>/ folders.

For each Pokemon folder:
  1. Verify every image actually opens (drop corrupt/truncated files).
  2. Drop images below a minimum resolution (thumbnails aren't useful for training).
  3. Drop near-duplicate images using perceptual hashing (average hash).
  4. Copy the surviving "clean" images into data/dataset/<pokemon>/.

Requires: pillow, imagehash
    pip install pillow imagehash --break-system-packages   (if on the sandbox)
    pip install pillow imagehash                            (normal environment)

Usage:
    python cleanup.py
    python cleanup.py --min-size 200 --hash-distance 5
"""

import os
import shutil
import argparse
import logging
from PIL import Image, UnidentifiedImageError
import imagehash

RAW_DIR = os.path.join("data", "raw")
DATASET_DIR = os.path.join("data", "dataset")
LOG_DIR = os.path.join("data", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("cleanup")
logger.setLevel(logging.INFO)
logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

file_handler = logging.FileHandler(os.path.join(LOG_DIR, "cleanup_log.txt"))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def clean_pokemon_folder(pokemon, min_size=150, hash_distance=5):
    """
    Cleans data/raw/<pokemon>/ and copies survivors to data/dataset/<pokemon>/.
    Returns a dict of counts for logging/summary purposes.
    """
    src_dir = os.path.join(RAW_DIR, pokemon)
    dst_dir = os.path.join(DATASET_DIR, pokemon)
    os.makedirs(dst_dir, exist_ok=True)

    if not os.path.isdir(src_dir):
        logger.warning(f"No raw folder found for '{pokemon}', skipping.")
        return None

    files = [f for f in os.listdir(src_dir) if f.lower().endswith(IMAGE_EXTENSIONS)]

    total = len(files)
    corrupt = 0
    too_small = 0
    duplicate = 0
    kept = 0

    seen_hashes = []  # list of imagehash objects already accepted

    for fname in files:
        fpath = os.path.join(src_dir, fname)

        # 1. Verify image opens and isn't corrupt
        try:
            with Image.open(fpath) as img:
                img.verify()  # checks integrity without fully loading
            # re-open after verify() (verify() leaves file unusable for further ops)
            with Image.open(fpath) as img:
                width, height = img.size
                img_converted = img.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as e:
            corrupt += 1
            logger.info(f"  [corrupt]   {pokemon}/{fname}: {e}")
            continue

        # 2. Filter by minimum resolution
        if width < min_size or height < min_size:
            too_small += 1
            logger.info(f"  [too small] {pokemon}/{fname}: {width}x{height}")
            continue

        # 3. Perceptual hash dedupe
        try:
            phash = imagehash.average_hash(img_converted)
        except Exception as e:
            corrupt += 1
            logger.info(f"  [hash fail] {pokemon}/{fname}: {e}")
            continue

        is_duplicate = any((phash - existing) <= hash_distance for existing in seen_hashes)
        if is_duplicate:
            duplicate += 1
            logger.info(f"  [duplicate] {pokemon}/{fname}")
            continue

        seen_hashes.append(phash)

        # Survived all checks -> copy to dataset dir
        dst_path = os.path.join(dst_dir, fname)
        shutil.copy2(fpath, dst_path)
        kept += 1

    summary = {
        "total": total,
        "corrupt": corrupt,
        "too_small": too_small,
        "duplicate": duplicate,
        "kept": kept,
    }

    logger.info(
        f"{pokemon:20s} total={total:4d}  kept={kept:4d}  "
        f"corrupt={corrupt:3d}  too_small={too_small:3d}  duplicate={duplicate:3d}"
    )
    return summary


def clean_all(min_size=150, hash_distance=5):
    if not os.path.isdir(RAW_DIR):
        logger.error(f"No raw data directory found at {RAW_DIR}")
        return

    pokemon_folders = sorted(
        d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))
    )

    logger.info(f"========== CLEANUP START ({len(pokemon_folders)} folders) ==========")
    overall = {"total": 0, "corrupt": 0, "too_small": 0, "duplicate": 0, "kept": 0}

    for pokemon in pokemon_folders:
        result = clean_pokemon_folder(pokemon, min_size=min_size, hash_distance=hash_distance)
        if result:
            for k in overall:
                overall[k] += result[k]

    logger.info("========== CLEANUP SUMMARY ==========")
    logger.info(
        f"TOTAL: total={overall['total']}  kept={overall['kept']}  "
        f"corrupt={overall['corrupt']}  too_small={overall['too_small']}  "
        f"duplicate={overall['duplicate']}"
    )
    logger.info("======================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and dedupe scraped Pokemon plush images.")
    parser.add_argument("--min-size", type=int, default=150, help="Minimum width/height in pixels.")
    parser.add_argument(
        "--hash-distance",
        type=int,
        default=5,
        help="Max perceptual hash distance to consider two images duplicates (lower = stricter).",
    )
    args = parser.parse_args()

    clean_all(min_size=args.min_size, hash_distance=args.hash_distance)
