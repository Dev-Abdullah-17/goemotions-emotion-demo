# Fixing the Long Tail in Fine-Grained Emotion Classification

A multi-label emotion classifier built on **DeBERTa-v3** and **GoEmotions** (28 emotions). The project isn't "fine-tune a transformer and report a score" — it's a diagnosis. GoEmotions has a severe class-imbalance problem that quietly wrecks the rare emotions, and this repo tracks a series of controlled experiments that identify that problem, fix the largest part of it, and honestly report one idea that *didn't* help.

**Headline result:** Macro-F1 **0.4769 → 0.5302** (+5.3 points), driven almost entirely by reviving emotions the baseline had given up on — `relief` and `grief` went from **0.00** to **0.35–0.44**.

🔗 **Live demo:** https://goemotions-emotion-demo-cksccxfpdawgisu4esdrzu.streamlit.app/
🤗 **Model:** https://huggingface.co/dev-abdullah-0909/goemotions-deberta-v3-asl

## The problem

GoEmotions labels 58k Reddit comments across 27 emotions plus `neutral`, and a comment can carry several at once (multi-label). Two properties make it hard, and most write-ups ignore both:

- **Extreme class imbalance.** In the test set, `neutral` has 1,787 examples; `grief` has **6**, `relief` **11**, `pride` **16**, `nervousness` **23**. A model trained with a standard loss sees the rare emotions so infrequently that the cheapest way to lower its loss is to *never predict them*.
- **Subjective labels.** Human annotators disagree a lot on emotion — inter-annotator agreement on GoEmotions is low by design, because emotion perception is genuinely subjective. This puts a real ceiling on achievable performance.

The consequence: a naive model posts a respectable-looking overall score while completely failing on a third of the emotion classes. This project measures that failure and attacks it.

## Why Macro-F1 is the headline metric

Micro-F1 pools every prediction together, so the handful of common emotions dominate it — a model can score well on Micro-F1 while ignoring every rare class. **Macro-F1 averages the per-class F1 scores, so `grief` counts as much as `neutral`.** It's the metric that actually punishes the imbalance problem, so it's the one reported first throughout.

## Approach: a controlled ablation

Each milestone changes exactly one thing from the previous one, so every change in the score is attributable.

| Milestone | Change | Macro-F1 | Micro-F1 |
|---|---|---|---|
| **M1** | DeBERTa-v3-base, plain BCE loss (baseline) | 0.4769 | 0.6041 |
| **M2** | + Asymmetric Loss (`gamma_neg=6`) | **0.5302** | 0.6023 |
| **M3** | + multi-label supervised contrastive (`lambda=0.1`) | 0.5238 | **0.6094** |

**M2 is the final model.** M3 is reported because a tested-and-rejected idea is part of the story (see below).

## Where the gains came from

The M1 → M2 jump is not spread evenly — it lands exactly where the imbalance problem lives:

| Emotion | support | M1 (BCE) | M2 (ASL) |
|---|---|---|---|
| grief | 6 | 0.00 | 0.35 |
| relief | 11 | 0.00 | 0.31 |
| pride | 16 | 0.11 | 0.39 |
| nervousness | 23 | 0.18 | 0.39 |
| realization | 145 | 0.18 | 0.25 |

Meanwhile the common emotions held or improved (`gratitude` 0.92, `amusement` 0.83, `love` 0.80). Macro-F1 rose **without** trading away the head of the distribution — which is the whole point.

## Key decisions and what they cost

**Asymmetric Loss over plain BCE.** BCE penalises a missed positive and a false alarm equally and weights every example the same, so rare-class errors barely register. Asymmetric Loss (Ben-Baruch et al., 2021) down-weights easy negatives and penalises missed positives harder — directly countering the "predict absent" shortcut. Tuning `gamma_neg` from 4 to 6 gave a small further gain, but with a caveat worth noting: **at `gamma_neg=6` the model is worse-calibrated** (Macro-F1 only 0.40 at a flat 0.5 threshold) and only overtakes `gamma_neg=4` *after* per-class threshold tuning. The loss and the decision threshold interact; reporting only the flat-0.5 number would have hidden this.

**Per-class threshold tuning.** A single 0.5 cutoff is almost never optimal across 28 imbalanced classes. Thresholds are optimised per class on the validation set and then applied unchanged to test. This alone contributed several points of Macro-F1 and is the reason the tuned thresholds land between 0.55 and 0.75 for most classes.

**fp16 disabled.** DeBERTa-v3 diverges to NaN under fp16 mixed precision (a documented quirk of the architecture), and the T4 doesn't support bf16 well. Training runs in fp32 — slower, but stable. A pre-training sanity check (untrained model should output loss ≈ 0.69, not NaN) gates every run.

## What didn't work — and why (M3)

The most interesting negative result. Emotions co-occur in patterns (`gratitude`+`admiration`, `grief`+`sadness`), so M3 added a **multi-label supervised contrastive loss**: a projection head shapes the embedding space so comments sharing emotions sit close together, weighted by the Jaccard overlap of their label sets. The classifier then reads from a representation that already encodes co-occurrence.

It didn't beat M2. Tested at `lambda` ∈ {0.1, 0.2}, both slightly *lowered* Macro-F1 (−0.006 to −0.010).

The likely reason is structural, not a bug: **contrastive learning is batch-hungry.** It needs many shared-label pairs per batch to form a strong signal, and a T4 caps the batch at 32. With 28 sparse labels, most batches contain too few positive pairs, so the auxiliary term mostly added noise that competed with the classification objective.

One genuine wrinkle: M3 posted the **best Micro-F1 of all three runs** (0.6094) while losing on Macro-F1. So the contrastive structure slightly helped the common/overall picture at the expense of the rare-class average — a real, nameable trade-off rather than a flat failure.

## Honest limitations

- **`grief` (support = 6) is effectively unlearnable** on this split. Its F1 swings between 0.00 and 0.35 depending on a single prediction. No method here fixes that, because the problem is data quantity, not modelling.
- The subjectivity ceiling (low inter-annotator agreement) caps how high *any* model can go on GoEmotions; ~0.53 Macro-F1 is competitive but the dataset itself is noisy.
- Evaluation is on the fixed GoEmotions test split, not cross-validated, so per-class numbers on tiny classes carry real variance.

## Repo structure

| File | Purpose |
|---|---|
| `emotion_m1_baseline.py` | Baseline: DeBERTa-v3 + BCE, threshold tuning, per-class report |
| `emotion_m2_asl.py` | + Asymmetric Loss (the final model) |
| `emotion_m3_contrastive.py` | + supervised contrastive loss (the rejected experiment) |

## Reproducing

Built for a single GPU (Kaggle T4). Each script is self-contained.

```bash
pip install "transformers==4.44.2" datasets sentencepiece scikit-learn accelerate
python emotion_m2_asl.py     # the final model
```

Notes that save hours: `transformers==4.44.2` loads DeBERTa-v3 cleanly for classification (newer builds mangled the head initialisation into NaNs); keep `fp16=False`; pin to a single GPU with `CUDA_VISIBLE_DEVICES=0` to avoid a multi-GPU / mixed-precision clash.

## What I'd try with more compute

- Larger batches (256+) to give the contrastive objective a fair test — the M3 result is a "not at this batch size" verdict, not a final one.
- Training on the **soft label distributions** GoEmotions provides (individual annotator votes) instead of majority-vote hard labels, and reporting calibration — turning the subjectivity problem from a limitation into the object of study.
- Distribution-balanced loss as a third point of comparison against ASL.
