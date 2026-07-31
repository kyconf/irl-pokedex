# Training Tier — Learning Notes

Companion to `split_data.py` and `train.py`. Those files carry the detailed
reasoning inline; this is the map, the vocabulary bridge, and the set of
experiments worth running yourself.

---

## Where this sits in the pipeline

```
scraper.py    -> data/raw/<class>/        scraped images, unvetted
cleanup.py    -> data/dataset/<class>/    verified, deduped, size-filtered
split_data.py -> data/splits/train|val/   held-out evaluation set        <- new
train.py      -> models/pokedex_best.pt   weights + class names          <- new
```

`split_data.py` writes to `data/splits/` rather than nesting inside
`data/dataset/`, so your cleaned pool stays a single source of truth and you
can re-split it any number of ways without re-running cleanup.

Run order:

```bash
pip install torch torchvision      # neither is in .venv yet
python cleanup.py                  # data/dataset/ is currently empty
python split_data.py
python train.py
```

---

## Theory → code

You know the concepts; this is where each one physically lives.

| Concept | Where it is |
|---|---|
| Forward pass | `outputs = model(inputs)` in `run_epoch` |
| Loss function | `nn.CrossEntropyLoss(label_smoothing=0.1)` |
| Backprop | `loss.backward()` — builds nothing itself, walks the graph autograd recorded during the forward pass |
| Gradient descent step | `optimizer.step()` |
| Learning rate | `head_lr` / `finetune_lr`, and the per-group `lr` in Phase B |
| Regularization | weight decay, dropout (already in MobileNetV3's head), label smoothing, augmentation, early stopping via best-checkpoint |
| Epoch vs batch | outer `for epoch`, inner `for inputs, labels in loader` |
| Train/eval mode | `model.train()` / `model.eval()` — changes Dropout and BatchNorm only |
| Overfitting | the train-acc-minus-val-acc gap in the logs |

Two things that surprise people coming from theory:

**There is no softmax in the model.** `CrossEntropyLoss` applies log-softmax
internally. The model emits *logits*. You apply softmax yourself at inference
time when you want probabilities for the UI. Doing it in both places is a real
bug that quietly flattens your gradients.

**`optimizer.zero_grad()` is not boilerplate.** PyTorch accumulates into
`.grad` by default. Omit the zeroing and each step is polluted by every step
before it. The symptom is a loss that just refuses to come down, with nothing
obviously wrong anywhere.

---

## The one idea that matters most here

Your project's actual research problem is **domain shift**: train distribution
(eBay product shots, white backdrop, studio light) ≠ deployment distribution
(your carpet, your lamp, your phone).

Three things in this code exist because of it, and they're the interesting
part of the project to talk about:

1. **Source-aware splitting** (`--source-aware`) — validate on your own photos
   only, so the metric measures the thing you care about instead of measuring
   how well the model memorized eBay's photography conventions.
2. **Aggressive color and crop augmentation** — makes background and lighting
   unusable as a shortcut, forcing the model onto the object itself.
3. **Partial unfreezing** — lets the late, domain-specific layers adapt to
   plush textures while the generic early layers stay put.

If someone asks what was hard about this project, this is the answer. Not the
model architecture — that's four lines.

---

## Reading your first run

Compare against `1/num_classes` = 0.167 for 6 classes. That's the
random-guessing floor.

At ~150 images expect roughly: Phase A val accuracy 0.5–0.7, Phase B pushing
0.65–0.8, with train accuracy far above both. The gap **is** overfitting, it
**is** expected at this data size, and the fix is data rather than
hyperparameters.

Read the confusion matrix at the end before touching anything. It tells you
which specific pairs the model confuses, which tells you what to photograph
next. That beats scraping 200 more images of a class that was already fine.

---

## Experiments worth running

Learning happens here, not from reading. Each of these is a one-flag change,
and the point is to predict the outcome first, then check.

1. **Delete the augmentation.** Replace `train_tf` with `val_tf` and retrain.
   Train accuracy will rocket toward 1.0 while val stagnates or drops. This is
   the cleanest demonstration of overfitting you'll ever get, on your own data.

2. **Skip Phase A** (`--head-epochs 0`). Watch the first few epochs. The
   random head's gradients wreck the pretrained features before they stabilize,
   and final accuracy comes out lower. This is the empirical justification for
   the two-phase schedule.

3. **Raise the fine-tune lr to 1e-2** (`--finetune-lr 1e-2`). Loss diverges or
   flatlines. Seeing a learning rate that's too large actually blow up builds
   much better intuition than reading that it can.

4. **Unfreeze everything** (`--unfreeze-blocks 17`). At 150 images this
   overfits harder. Re-run it after you get to 200+ images per class — it
   should then start to *help*. That crossover is the bias-variance tradeoff
   showing up in your own logs.

5. **Compare a random split to a source-aware split.** Same model, same
   hyperparameters. The random split will report a meaningfully higher number
   on the same underlying model. Sit with that — it's the most important
   lesson in the whole project.

---

## What to do next, in priority order

1. **More data.** 25/class is the binding constraint; everything else is
   secondary. Video frame extraction is the cheap path: a 30-second clip at
   2fps is 60 frames per plushie. Rotate the toy, change rooms, change
   lighting, vary distance. Name them `vid_<pokemon>_0001.jpg` so
   `--source-aware` picks them up automatically.

2. **Then re-run the experiments above.** Several conclusions flip once you
   have real data volume, and watching them flip is the point.

3. **Then the API tier.** Load the checkpoint, read `class_names` from it (do
   not hardcode the list — the ordering is load-bearing), apply the *val*
   transform exactly, softmax the logits for a confidence score.

4. **Consider a "not a plushie" path.** Right now the model must answer with
   one of 6 classes; point it at a coffee mug and it will confidently say
   Gengar. A confidence threshold on the softmax output is the quick fix, and
   it's where label smoothing earns its keep, since it keeps those
   probabilities better calibrated.

---

## Bugs to expect

- **Forgot `model.eval()` at validation** → val accuracy inexplicably bad.
  Dropout is still firing and BatchNorm is using noisy batch statistics.
- **Frozen backbone still drifting** → froze `requires_grad` but left the
  module in train mode, so BatchNorm running stats kept updating. See
  `set_train_mode()`.
- **Class names out of order in the API** → always read them from the
  checkpoint, never rebuild the list independently.
- **MPS out of memory or odd errors** → drop `--batch-size` to 16, or run on
  CPU. At this dataset size CPU training still finishes in minutes.
- **`num_workers` hangs on macOS** → set `--num-workers 0`.
