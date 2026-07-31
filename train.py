"""
train.py

Fine-tunes MobileNetV3-Large on the plushie dataset produced by
split_data.py, and writes the best checkpoint to models/pokedex_best.pt.

    pip install torch torchvision
    python train.py
    python train.py --head-epochs 10 --finetune-epochs 15 --batch-size 32

Reads:  data/splits/train/<pokemon>/ and data/splits/val/<pokemon>/
Writes: models/pokedex_best.pt   (weights + class names + metadata)


===========================================================================
THE BIG PICTURE: WHY TRANSFER LEARNING, AND WHY IT WORKS
===========================================================================

You have ~150 images. A MobileNetV3-Large has about 5.4 million parameters.
Training that from scratch is hopeless — vastly more knobs than constraints,
so the network just memorizes the training set perfectly and learns nothing
that generalizes. The usual rule of thumb for training a CNN from random
initialization is on the order of a thousand images per class, and you have
twenty-five.

Transfer learning sidesteps this. A network pretrained on ImageNet (1.2M
images, 1000 classes) has already learned a hierarchy of visual features, and
crucially that hierarchy is mostly generic:

    early layers   -> edges, corners, color blobs, gradients
    middle layers  -> textures, fabric weave, fur, repeated patterns
    late layers    -> object parts, shapes, "ear-like thing", "round body"
    final layer    -> "this specific arrangement of parts = golden retriever"

Only that last bit is ImageNet-specific. Everything before it is just... how
images work. Edge detectors are edge detectors whether you're looking at a
retriever or a Gengar plush. So you keep the learned feature extractor, throw
away the 1000-class head, bolt on a fresh 6-class head, and train that. You've
reduced the problem from "learn vision" to "learn a linear map from 1280
already-meaningful numbers to 6 classes" — which 150 images CAN support.

Plush toys are an unusually good fit for this, incidentally: ImageNet contains
teddy bears and other stuffed toys, so the pretrained features already encode
"soft fabric object" texture statistics.


===========================================================================
THE TWO-PHASE SCHEDULE, AND THE GRADIENT ARGUMENT FOR IT
===========================================================================

PHASE A — freeze the backbone, train only the new head. ~10 epochs, lr 1e-3.

    The new head is randomly initialized, so on step 1 its outputs are noise
    and its loss is huge. Backprop sends that huge, meaningless gradient
    backwards through the whole network. If the backbone is unfrozen, those
    first few updates actively scramble pretrained weights that took days of
    GPU time to learn. You'd be destroying the thing you came for. So: freeze
    the backbone, let the head converge to something sane first.

    Bonus: with the backbone frozen there are no gradients to compute for
    95% of the network, so epochs are much faster.

PHASE B — unfreeze the last few blocks, lr 1e-4 with cosine decay. ~15 epochs.

    Now the head is sane, so gradients flowing back are meaningful, and we can
    let the late layers adapt. We unfreeze only features[-4:] — the last few
    blocks — because that's where the ImageNet-specific "object identity"
    representations live and where your domain differs most. The early edge
    detectors are already correct for your problem; there's nothing to gain by
    letting 150 images push them around, and plenty to lose.

    The learning rate drops 10x. Large steps are for finding the right basin;
    small steps are for settling into it. Take 1e-3 steps on pretrained
    weights and you'll walk straight out of the good solution.

This staged unfreezing is standard practice, and being able to explain WHY
(the random-head gradient argument) is the thing that separates understanding
it from having copied it.


===========================================================================
HOW TO READ THE OUTPUT — THE FOUR STORIES YOUR CURVES CAN TELL
===========================================================================

Each epoch prints train loss/acc and val loss/acc. The GAP between them is
the diagnostic:

  train acc high,  val acc high, small gap  -> working. ship it.
  train acc high,  val acc low,  huge gap   -> OVERFITTING. memorizing the
                                               training images. more data or
                                               stronger augmentation.
  train acc low,   val acc low              -> UNDERFITTING. not enough
                                               capacity or training. unfreeze
                                               more layers, train longer,
                                               raise lr.
  val acc > train acc                       -> normal here, not a bug. Train
                                               acc is measured with dropout on
                                               and heavy augmentation applied;
                                               val is measured clean. You're
                                               grading train on a harder exam.

Expect overfitting at 150 images. That's the honest baseline, and the fix is
more data (video frames), not more clever code.
"""

import os
import json
import copy
import argparse
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

SPLITS_DIR = os.path.join("data", "splits")
MODEL_DIR = "models"
LOG_DIR = os.path.join("data", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

logger = logging.getLogger("train")
logger.setLevel(logging.INFO)
logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

file_handler = logging.FileHandler(os.path.join(LOG_DIR, "train_log.txt"))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# ---------------------------------------------------------------------------
# ImageNet normalization constants.
#
# These are the per-channel mean and std of the ImageNet training set. You
# subtract the mean and divide by the std so each channel is roughly centered
# at 0 with unit variance.
#
# Why these exact numbers and not your own dataset's statistics? Because the
# pretrained weights were LEARNED on inputs normalized this way. Every filter
# in that network expects to see values in this distribution. Feed it raw
# 0-1 pixels instead and every activation is shifted from what the weights
# were tuned for — you'd be handing it inputs in the wrong units. Matching
# the pretraining preprocessing is mandatory, not stylistic.
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224   # what MobileNetV3 was pretrained at; deviating costs accuracy


def build_transforms(img_size=IMG_SIZE):
    """
    Two pipelines: aggressive random augmentation for training, deterministic
    resize+crop for validation.

    -------------------------------------------------------------------------
    WHAT AUGMENTATION ACTUALLY DOES
    -------------------------------------------------------------------------
    It is not "making more data" in any information-theoretic sense — a rotated
    Pikachu contains no new information about Pikachu. What it does is encode
    an INVARIANCE: it tells the model "these transformations do not change the
    label." Every epoch the model sees a differently-mangled version of the
    same image, so it can't lock onto any one incidental detail; the only
    signal that survives all the mangling is the actual object.

    That's the whole game for your project. With 25 eBay photos per class, the
    single easiest way for the model to separate the classes is background and
    lighting, not the plush itself. Augmentation is what makes that shortcut
    stop working.

    Choose augmentations that match variation you expect at inference time:

      RandomResizedCrop  - the strongest one here. Crops a random sub-region
                           (10%-100% of the image) and resizes to 224. Forces
                           the model to identify the toy from partial views and
                           at many scales, exactly like a phone held at varying
                           distance. Also destroys fixed framing, so "object
                           dead-center on white" stops being a usable cue.
      HorizontalFlip     - a mirrored plushie is still that plushie. Free 2x.
      Rotation(20 deg)   - you won't hold the phone perfectly level.
      ColorJitter        - warm indoor bulbs vs cool daylight vs studio white
                           balance. This one is doing real work for you: it's
                           what stops the model keying on "white background =
                           this scraped class."
      RandomErasing      - blanks out a random rectangle AFTER normalization.
                           Simulates occlusion and stops reliance on any single
                           region (e.g. only ever looking at the ears).

    Deliberately NOT included:
      VerticalFlip       - upside-down plushies aren't in your production
                           distribution. Teaching an invariance you'll never
                           need just makes the problem harder for no gain.
      Grayscale          - color is highly discriminative here (Gengar purple,
                           Charmander orange). Don't throw away your best cue.

    Validation gets NO random augmentation. Val must be deterministic or the
    metric changes run to run for reasons that have nothing to do with the
    model. Resize(256) then CenterCrop(224) is the standard ImageNet eval
    recipe, and matching it matters for the same reason the normalization
    constants do.
    """
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.ToTensor(),                       # PIL/HWC/0-255 -> tensor/CHW/0-1
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),   # after ToTensor by design
    ])

    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),     # 224 * 1.14 ~= 256
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_tf, val_tf


def pick_device():
    """
    MPS is Apple Silicon's GPU backend. CUDA is NVIDIA. CPU works, just slowly
    — though at your dataset size even CPU training finishes in minutes, so
    don't let device selection block you.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes, device):
    """
    Load pretrained MobileNetV3-Large and replace its classifier head.

    -------------------------------------------------------------------------
    THE SURGERY
    -------------------------------------------------------------------------
    The model has two parts:
        model.features    - the convolutional backbone (the useful bit)
        model.classifier  - Sequential(Linear(960,1280), Hardswish,
                                       Dropout(0.2), Linear(1280,1000))

    Index 3 is that final Linear mapping to ImageNet's 1000 classes. Swap it
    for Linear(1280, num_classes). The new layer is randomly initialized — it
    is the ONLY part of the network that knows nothing, which is exactly why
    Phase A exists.

    Note there's no Softmax at the end. That surprises people. It's because
    nn.CrossEntropyLoss applies log-softmax internally, and doing it twice is
    a bug. The raw outputs are called LOGITS: unbounded real numbers, one per
    class, where bigger means more confident. You only apply softmax manually
    at inference time, when you want probabilities to display in the Pokedex
    UI. Remember this when you write the FastAPI tier.
    """
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
    model = mobilenet_v3_large(weights=weights)

    in_features = model.classifier[3].in_features      # 1280
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model.to(device)


def set_backbone_trainable(model, trainable, last_n_blocks=None):
    """
    Freeze or unfreeze the backbone.

    -------------------------------------------------------------------------
    WHAT requires_grad ACTUALLY DOES
    -------------------------------------------------------------------------
    Setting p.requires_grad = False tells autograd not to accumulate a
    gradient for that tensor. No gradient means the optimizer has nothing to
    apply, so the weight is pinned. It also means autograd can skip building
    part of the computation graph, which is why frozen epochs run faster.

    -------------------------------------------------------------------------
    THE BATCHNORM TRAP  (this one bites everybody once)
    -------------------------------------------------------------------------
    BatchNorm layers hold two kinds of state:
        learnable    - weight/bias, controlled by requires_grad
        running stats- running_mean/running_var, NOT parameters, updated on
                       every forward pass whenever the module is in train mode

    So freezing requires_grad does NOT fully freeze a BatchNorm layer. Its
    running statistics keep drifting toward YOUR batch statistics — and with
    a batch size of 32 on 150 images, those estimates are noisy garbage
    compared to the ones estimated over ImageNet. Your "frozen" backbone
    quietly degrades and you get a mystifying accuracy drop.

    The fix is to also put the frozen module in eval() mode, which switches
    BatchNorm to using its stored running stats and stops updating them. See
    set_train_mode() below, which has to re-apply this after every model.train()
    call because model.train() recursively overrides every child module.
    """
    if trainable and last_n_blocks is not None:
        # Unfreeze only the last N blocks; keep early layers pinned.
        for p in model.features.parameters():
            p.requires_grad = False
        for block in model.features[-last_n_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
    else:
        for p in model.features.parameters():
            p.requires_grad = trainable

    # The head is always trainable.
    for p in model.classifier.parameters():
        p.requires_grad = True


def set_train_mode(model, backbone_frozen):
    """
    model.train() flips EVERY submodule into training mode, which re-enables
    BatchNorm running-stat updates in the backbone we just froze. So we call
    model.train() and then walk it back for the frozen part.
    """
    model.train()
    if backbone_frozen:
        model.features.eval()


def run_epoch(model, loader, criterion, optimizer, device, train, backbone_frozen=False):
    """
    One pass over a dataloader. Returns (average_loss, accuracy).

    -------------------------------------------------------------------------
    THE TRAINING LOOP, LINE BY LINE
    -------------------------------------------------------------------------
        optimizer.zero_grad()   PyTorch ACCUMULATES gradients into .grad by
                                default (useful for simulating large batches).
                                Forget this and batch 5's update is polluted by
                                batches 1-4. Classic silent bug: your loss just
                                mysteriously refuses to go down.

        outputs = model(x)      forward pass -> logits, shape [batch, classes]

        loss = criterion(...)   scalar measuring wrongness

        loss.backward()         backprop. Walks the graph autograd recorded
                                during the forward pass, computing d(loss)/d(w)
                                for every w with requires_grad=True.

        optimizer.step()        applies those gradients per the update rule.

    torch.set_grad_enabled(train) is the eval-side counterpart: during
    validation we don't need gradients at all, so we skip building the graph.
    Saves memory and time. (torch.no_grad() is the same idea with less
    ceremony; this form just lets one function serve both paths.)

    model.eval() vs model.train() changes behaviour of exactly two things:
    Dropout (active in train, identity in eval) and BatchNorm (batch stats in
    train, running stats in eval). Forgetting model.eval() at validation is
    the single most common source of "why is my val accuracy garbage."
    """
    if train:
        set_train_mode(model, backbone_frozen)
    else:
        model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    with torch.set_grad_enabled(train):
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            # loss is a per-sample MEAN, so weight by batch size before summing;
            # otherwise a final short batch would be over-weighted.
            running_loss += loss.item() * inputs.size(0)

            # argmax over the class dimension = predicted class index
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

    return running_loss / running_total, running_correct / running_total


def evaluate_per_class(model, loader, device, class_names):
    """
    Aggregate accuracy hides everything interesting. A model at 70% overall
    might be at 95% on Pikachu and 30% on Bulbasaur, and that tells you where
    to point the camera next.

    The confusion matrix goes further: it shows WHICH class the mistakes go
    to. Bulbasaur being read as Gengar is a real signal (both roundish, both
    blue-purple-ish under warm light) and suggests you need more varied
    lighting for those two specifically. Scraping 200 more Pikachu wouldn't
    have helped at all. This is how you decide what data to collect next
    instead of guessing.
    """
    model.eval()
    n = len(class_names)
    confusion = [[0] * n for _ in range(n)]

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            preds = model(inputs).argmax(dim=1).cpu()
            for true_idx, pred_idx in zip(labels.tolist(), preds.tolist()):
                confusion[true_idx][pred_idx] += 1

    logger.info("--- Per-class validation accuracy ---")
    for i, name in enumerate(class_names):
        total = sum(confusion[i])
        correct = confusion[i][i]
        pct = (correct / total * 100) if total else 0.0
        logger.info(f"  {name:20s} {correct:3d}/{total:3d}  ({pct:5.1f}%)")

    logger.info("--- Confusion matrix (rows = true, cols = predicted) ---")
    header = " " * 20 + "".join(f"{c[:6]:>8s}" for c in class_names)
    logger.info(header)
    for i, name in enumerate(class_names):
        row = f"{name:20s}" + "".join(f"{v:8d}" for v in confusion[i])
        logger.info(row)

    return confusion


def train_model(head_epochs, finetune_epochs, batch_size, head_lr, finetune_lr,
                unfreeze_blocks, num_workers):
    device = pick_device()
    logger.info(f"Device: {device}")

    train_dir = os.path.join(SPLITS_DIR, "train")
    val_dir = os.path.join(SPLITS_DIR, "val")
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        logger.error(f"Missing {train_dir} or {val_dir}. Run split_data.py first.")
        return

    train_tf, val_tf = build_transforms()

    # ImageFolder infers classes from subdirectory names, sorted alphabetically,
    # and assigns integer labels 0..N-1 in that order. That ordering is why we
    # save class_names into the checkpoint: if the FastAPI service reconstructs
    # the label list any other way and the order differs, every prediction is
    # silently mislabeled — the model is fine, your Pokedex just confidently
    # calls Gengar "Charmander". Persist the mapping WITH the weights.
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
    class_names = train_ds.classes

    logger.info(f"Classes ({len(class_names)}): {class_names}")
    logger.info(f"Train images: {len(train_ds)}   Val images: {len(val_ds)}")

    # shuffle=True on train matters: fixed ordering means every epoch produces
    # the same sequence of gradient steps, which correlates updates and can
    # stall optimization. Never shuffle val — you want identical conditions
    # every evaluation.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False)

    model = build_model(len(class_names), device)

    # -----------------------------------------------------------------------
    # LOSS: cross-entropy with label smoothing
    #
    # Cross-entropy pushes the logit of the correct class up and the rest down.
    # Minimizing it is equivalent to maximizing the likelihood of the correct
    # labels — the standard choice for single-label classification.
    #
    # label_smoothing=0.1 changes the target from "100% Pikachu, 0% everything
    # else" to "90% Pikachu, 10% spread over the rest." Hard targets push the
    # model toward infinite logits and total confidence, which on a small
    # dataset means confident memorization AND badly calibrated probabilities.
    # Since your UI displays a confidence score, calibration is a real feature,
    # not a nicety — you want "60% sure" to mean roughly 60% right.
    # -----------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # -1.0 rather than 0.0 so the first epoch always beats it. Starting at 0.0
    # means a run where val accuracy is 0.0 throughout never records a
    # checkpoint at all, and we'd try to save state_dict=None.
    best_acc = -1.0
    best_state = None
    history = []

    # =======================================================================
    # PHASE A - frozen backbone, train the head only
    # =======================================================================
    logger.info("=" * 60)
    logger.info(f"PHASE A: training head only ({head_epochs} epochs, lr={head_lr})")
    logger.info("=" * 60)

    set_backbone_trainable(model, trainable=False)

    # Only hand the optimizer parameters that actually require grad. Passing
    # frozen params is wasteful, and with optimizers that have weight decay it
    # can even nudge frozen weights via the decay term.
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=head_lr,
        weight_decay=1e-4,   # L2-ish penalty on weight magnitude; mild regularizer
    )

    for epoch in range(1, head_epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer,
                                    device, train=True, backbone_frozen=True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer,
                                    device, train=False)

        logger.info(
            f"[A {epoch:2d}/{head_epochs}] "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.3f}"
        )
        history.append({"phase": "A", "epoch": epoch, "train_loss": tr_loss,
                        "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})

        if va_acc > best_acc:
            best_acc = va_acc
            # deepcopy, because the live model keeps training and would mutate
            # any reference we held. Also move to CPU so the checkpoint isn't
            # pinned to a device that may not exist on the machine that loads it.
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})

    # =======================================================================
    # PHASE B - unfreeze the last blocks, lower lr, cosine decay
    # =======================================================================
    logger.info("=" * 60)
    logger.info(f"PHASE B: fine-tuning last {unfreeze_blocks} blocks "
                f"({finetune_epochs} epochs, lr={finetune_lr})")
    logger.info("=" * 60)

    set_backbone_trainable(model, trainable=True, last_n_blocks=unfreeze_blocks)

    # Discriminative learning rates: the newly-unfrozen backbone blocks get a
    # smaller lr than the head. The head is still the least-settled part of the
    # network and can take bigger steps; the pretrained conv blocks hold
    # information worth preserving, so they get gentler updates. Same idea as
    # the phase split, expressed per-parameter-group.
    backbone_params = [p for p in model.features.parameters() if p.requires_grad]
    head_params = list(model.classifier.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": finetune_lr},
        {"params": head_params, "lr": finetune_lr * 10},
    ], weight_decay=1e-4)

    # Cosine annealing decays lr smoothly from its initial value toward ~0 over
    # the phase. Rationale: early on you want to move; late on you want to stop
    # bouncing around the minimum and settle. A constant lr keeps rattling and
    # your final epochs are noise.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs)

    for epoch in range(1, finetune_epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer,
                                    device, train=True, backbone_frozen=False)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer,
                                    device, train=False)
        scheduler.step()   # once per EPOCH, not per batch, given T_max is in epochs

        logger.info(
            f"[B {epoch:2d}/{finetune_epochs}] "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.3f} | "
            f"lr {scheduler.get_last_lr()[0]:.2e}"
        )
        history.append({"phase": "B", "epoch": epoch, "train_loss": tr_loss,
                        "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})

        if va_acc > best_acc:
            best_acc = va_acc
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})

    # =======================================================================
    # Save the BEST checkpoint, not the last one.
    #
    # Val accuracy is not monotonic — it typically peaks and then declines as
    # the model starts overfitting. Keeping the best-so-far weights is
    # "early stopping" without actually halting training: you still see the
    # full curve, but you keep the good weights. Cheap, and it costs you
    # nothing but memory.
    #
    # Honest caveat, since you're learning this properly: selecting the
    # checkpoint by val accuracy means val has now influenced a decision, so
    # it's no longer a fully unbiased estimate of true performance. With more
    # data the rigorous move is three splits — train / val (for decisions) /
    # test (touched exactly once, at the end). At 150 images you don't have
    # the data to afford it, but you should know that's the shortcut you're
    # taking and be able to say so.
    # =======================================================================
    if best_state is not None:
        model.load_state_dict(best_state)

    model.to(device)
    evaluate_per_class(model, val_loader, device, class_names)

    ckpt_path = os.path.join(MODEL_DIR, "pokedex_best.pt")
    torch.save({
        "state_dict": best_state,
        "class_names": class_names,      # ordering is load-bearing, see above
        "arch": "mobilenet_v3_large",
        "img_size": IMG_SIZE,
        "norm_mean": IMAGENET_MEAN,      # so the API preprocesses identically
        "norm_std": IMAGENET_STD,
        "val_acc": best_acc,
    }, ckpt_path)

    with open(os.path.join(LOG_DIR, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Best validation accuracy: {best_acc:.3f}")
    logger.info(f"Checkpoint saved to {ckpt_path}")
    logger.info(f"Baseline for reference: random guessing = {1 / len(class_names):.3f}")
    logger.info("=" * 60)

    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune MobileNetV3 on plushie images.")
    parser.add_argument("--head-epochs", type=int, default=10, help="Phase A epochs (frozen backbone).")
    parser.add_argument("--finetune-epochs", type=int, default=15, help="Phase B epochs (partial unfreeze).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-lr", type=float, default=1e-3, help="Phase A learning rate.")
    parser.add_argument("--finetune-lr", type=float, default=1e-4, help="Phase B backbone learning rate.")
    parser.add_argument("--unfreeze-blocks", type=int, default=4, help="How many trailing blocks to unfreeze.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes.")
    args = parser.parse_args()

    train_model(
        head_epochs=args.head_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        head_lr=args.head_lr,
        finetune_lr=args.finetune_lr,
        unfreeze_blocks=args.unfreeze_blocks,
        num_workers=args.num_workers,
    )
