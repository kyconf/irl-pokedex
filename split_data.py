"""
split_data.py

Turns data/dataset/<pokemon>/ (flat, one folder per class) into a
train/val split that PyTorch's ImageFolder can read directly:

    data/splits/train/<pokemon>/...
    data/splits/val/<pokemon>/...

Run this AFTER cleanup.py.

    python split_data.py
    python split_data.py --val-frac 0.2 --seed 42
    python split_data.py --source-aware      # hold out your own photos as val


===========================================================================
WHY THIS SCRIPT EXISTS AT ALL  (the concept, not the code)
===========================================================================

A model's training loss tells you almost nothing. A model can drive training
loss to zero by memorizing its training images and still be useless. The only
number that means anything is performance on data the model has never had a
gradient step taken on. That is what the validation split is: a held-out
sample you use as a stand-in for "the future images this thing will actually
see in production."

That framing has a consequence most tutorials skip: a validation set is only
a good stand-in if it's drawn from the same distribution as production. Your
production distribution is "photos Kyle takes of his own plushies in his own
room with his own phone." Your training data is mostly eBay product shots on
white backgrounds. Those are different distributions. This is the core
problem your project is actually about — it's called DOMAIN SHIFT, and it is
the reason a random split will lie to you.

Concretely, here's the failure mode. Suppose you scrape 25 eBay photos of
Gengar and randomly put 20 in train and 5 in val. Those 5 val images look
exactly like the 20 training ones: same white backdrop, same studio lighting,
same catalog framing. The model can score 95% on that val set by learning
"purple blob on white background = Gengar" — and then collapse to 40% on your
carpet. Your val number said 95%. It was measuring the wrong thing. You would
have shipped a broken model and had no signal that anything was wrong.

--source-aware fixes this by putting ALL your own captured photos in val and
ALL the scraped photos in train. Now val accuracy answers the question you
actually care about: "does this work on my plushies, in my room?" It will be
a much lower and much uglier number. That is the point. A pessimistic honest
metric beats an optimistic fake one, because you make decisions from it.

(This is also a strong interview answer. "How did you split your data?" ->
"Source-aware, because a random split leaks background statistics between
train and val and inflates the metric" is a much better answer than "80/20.")


===========================================================================
THE OTHER RULE THIS SCRIPT ENFORCES: SPLIT BEFORE YOU TOUCH ANYTHING
===========================================================================

The split happens here, on raw files, before any augmentation, normalization,
or resizing. If you augment first and split second, augmented copies of the
SAME source image land on both sides of the split — the model sees a rotated
version of a val image during training. That's DATA LEAKAGE, and it silently
inflates your val score. Rule of thumb: split first, always. Anything derived
from the data gets computed on the training half only.

Note that cleanup.py's perceptual-hash dedupe is doing related work: two
near-identical scrapes of the same eBay listing, one in train and one in val,
is the same leak wearing a different hat.
"""

import os
import shutil
import random
import argparse
import logging

# Same directory/logging conventions as scraper.py and cleanup.py.
DATASET_DIR = os.path.join("data", "dataset")   # input:  cleaned, flat by class
SPLITS_DIR = os.path.join("data", "splits")     # output: train/ and val/ by class
LOG_DIR = os.path.join("data", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("split")
logger.setLevel(logging.INFO)
logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

file_handler = logging.FileHandler(os.path.join(LOG_DIR, "split_log.txt"))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# Filename convention for "photos I took myself" (video frames, phone pics).
# icrawler names its downloads 000001.jpg, 000002.jpg, ... so anything you
# capture yourself should be prefixed to distinguish it. When you write the
# video frame extractor, have it emit  vid_<pokemon>_0001.jpg  and this
# script's --source-aware mode will pick those up automatically.
OWN_PHOTO_PREFIXES = ("vid_", "own_", "irl_")


def is_own_photo(filename):
    """True if this image came from your own camera rather than a scrape."""
    return filename.lower().startswith(OWN_PHOTO_PREFIXES)


def split_class(pokemon, val_frac, rng, source_aware):
    """
    Split one class folder into train/val file lists.

    Returns (train_files, val_files, mode_used).
    """
    src_dir = os.path.join(DATASET_DIR, pokemon)
    files = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(IMAGE_EXTENSIONS))

    if not files:
        return [], [], "empty"

    own = [f for f in files if is_own_photo(f)]
    scraped = [f for f in files if not is_own_photo(f)]

    # --- Source-aware mode: your photos become val, scrapes become train ---
    # We only do this if there are actually enough of your own photos to make
    # a meaningful val set. Validating on 2 images is noise: each image is
    # worth 50 percentage points, so the metric jumps around wildly and tells
    # you nothing about whether a change helped.
    if source_aware and len(own) >= 5:
        logger.info(f"  {pokemon}: source-aware split ({len(scraped)} scraped -> train, {len(own)} own -> val)")
        return scraped, own, "source-aware"

    if source_aware:
        logger.warning(
            f"  {pokemon}: only {len(own)} own-photo(s) found (need >=5), "
            f"falling back to random split. Capture more video frames for an honest val set."
        )

    # --- Random stratified split ---
    # "Stratified" = the split is done per class, so every class keeps roughly
    # the same train/val ratio. If you split the pooled file list globally
    # instead, random chance can hand you a val set with 9 Pikachu and 0
    # Gengar, and your val accuracy becomes mostly a measurement of Pikachu.
    #
    # We shuffle with a SEEDED rng so this is reproducible: same seed, same
    # split, every run. Without that, every re-run reshuffles which images are
    # held out, and you can't tell whether a val-accuracy change came from
    # your code change or from getting an easier val set this time.
    shuffled = files[:]          # copy: don't mutate the caller's list
    rng.shuffle(shuffled)

    n_val = max(1, round(len(shuffled) * val_frac))
    val_files = shuffled[:n_val]
    train_files = shuffled[n_val:]

    logger.info(f"  {pokemon}: random split ({len(train_files)} train, {len(val_files)} val)")
    return train_files, val_files, "random"


def copy_files(pokemon, filenames, split_name):
    """Copy a list of files from data/dataset/<pokemon>/ into data/splits/<split>/<pokemon>/."""
    src_dir = os.path.join(DATASET_DIR, pokemon)
    dst_dir = os.path.join(SPLITS_DIR, split_name, pokemon)
    os.makedirs(dst_dir, exist_ok=True)

    for fname in filenames:
        shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))


def build_splits(val_frac=0.2, seed=42, source_aware=False):
    if not os.path.isdir(DATASET_DIR):
        logger.error(f"No dataset directory at {DATASET_DIR}. Run cleanup.py first.")
        return

    classes = sorted(
        d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))
    )
    if not classes:
        logger.error(f"{DATASET_DIR} exists but has no class folders. Run cleanup.py first.")
        return

    # Wipe any previous split so re-running doesn't leave stale files behind.
    # Without this, changing --seed would leave the old split's images sitting
    # in val/ alongside the new ones, and an image could end up in BOTH train
    # and val. That's leakage introduced by your own tooling, which is a
    # genuinely nasty bug to track down.
    if os.path.isdir(SPLITS_DIR):
        # onerror handler: copy2 preserves the source file's permission bits, so
        # if a scraped file landed read-only the delete fails. Chmod and retry.
        def _force_remove(func, path, _exc_info):
            os.chmod(path, 0o700)
            func(path)

        shutil.rmtree(SPLITS_DIR, onerror=_force_remove)

    # A dedicated rng instance rather than the global random module, so this
    # function's behaviour can't be perturbed by seeding done elsewhere.
    rng = random.Random(seed)

    logger.info(f"========== SPLIT START ({len(classes)} classes, seed={seed}) ==========")

    totals = {"train": 0, "val": 0}
    per_class = {}

    for pokemon in classes:
        train_files, val_files, mode = split_class(pokemon, val_frac, rng, source_aware)
        if mode == "empty":
            logger.warning(f"  {pokemon}: no images, skipping.")
            continue

        copy_files(pokemon, train_files, "train")
        copy_files(pokemon, val_files, "val")

        totals["train"] += len(train_files)
        totals["val"] += len(val_files)
        per_class[pokemon] = (len(train_files), len(val_files))

    logger.info("========== SPLIT SUMMARY ==========")
    for pokemon, (n_train, n_val) in per_class.items():
        logger.info(f"{pokemon:20s} train={n_train:4d}  val={n_val:4d}")
    logger.info(f"{'TOTAL':20s} train={totals['train']:4d}  val={totals['val']:4d}")

    # --- Class balance warning -------------------------------------------
    # If one class has 4x the images of another, the model can lower its
    # average loss just by guessing the big class more often. Accuracy then
    # flatters you: with 90% Pikachu, a model that ALWAYS says "Pikachu"
    # scores 90% while having learned nothing. Worth knowing before you
    # celebrate a number.
    if per_class:
        counts = {k: v[0] for k, v in per_class.items()}
        biggest = max(counts.values())
        smallest = min(counts.values())
        if smallest > 0 and biggest / smallest > 2.0:
            logger.warning(
                f"Class imbalance: largest training class is {biggest / smallest:.1f}x the smallest. "
                f"Consider scraping more of the small classes, or pass class weights to the loss."
            )

    # --- Sample size warning ---------------------------------------------
    if totals["val"] < 30:
        logger.warning(
            f"Only {totals['val']} validation images total. Each one is worth "
            f"{100 / max(totals['val'], 1):.1f} percentage points of accuracy, so the metric "
            f"will be very noisy. Treat small differences between runs as meaningless."
        )

    logger.info("===================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split cleaned dataset into train/val folders.")
    parser.add_argument("--val-frac", type=float, default=0.2, help="Fraction held out for validation.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed, for a reproducible split.")
    parser.add_argument(
        "--source-aware",
        action="store_true",
        help="Hold out your own captured photos (vid_/own_/irl_ prefixes) as validation.",
    )
    args = parser.parse_args()

    build_splits(val_frac=args.val_frac, seed=args.seed, source_aware=args.source_aware)
