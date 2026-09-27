# v2 pre-registration — mask-guided training

*Written and committed **before** any v2 model is trained. Nothing in this plan may be changed after v2 results are seen.*

## Motivation (from earlier steps)
- **Step 7 (Grad-CAM):** false alarms were driven by regions outside the lesion (below the lesion, image borders).
- **Step 8 (BUS-BRA):** AUC dropped from 0.96–0.98 (BUSI test) to 0.72–0.75 (external). BUS-BRA images are mostly tall, narrow crops, unlike BUSI.

v2 is motivated by these observations. BUS-BRA results will **not** be used to tune anything.

## Variants (both fixed in advance)
| Name | Change vs v1 (Step 5/6 baseline DenseNet121) |
|---|---|
| **v2a** | + auxiliary lesion-mask loss: a 1×1 conv on the final 7×7 feature map predicts the lesion mask (downsampled to 7×7); loss = cross-entropy + **λ·BCE**, **λ = 0.5** (fixed, not tuned) |
| **v2b** | v2a + **aspect-ratio crop augmentation**: with p = 0.5, crop the training image to a random width of 50–100% and height of 80–100%, always keeping the whole lesion inside the crop |

Everything else is identical to v1: 224px letterbox, ImageNet-pretrained DenseNet121, AdamW lr 1e-4, cosine schedule, class weights, batch 32, ≤30 epochs, early stopping on validation AUC (patience 7), seed 42, same 5 folds.

## Evaluation protocol
1. Train each variant with 5-fold CV on BUSI trainval (same folds as v1).
2. **Selection rule (BUSI only):** the final v2 is the variant with the higher pooled out-of-fold AUC. If the two differ by < 0.005, choose the simpler **v2a**.
3. Freeze each variant's threshold from its own OOF predictions (sensitivity ≥ 0.90), as in Step 6.
4. Evaluate v1, v2a and v2b **once** on the BUSI test set and **once** on BUS-BRA (all 1,875 images), no re-tuning.
5. Primary comparison: ensemble AUC on BUS-BRA, **selected v2 vs v1**, with a patient-level paired bootstrap (2,000 resamples). v2 is called an improvement only if the 95% CI of ΔAUC excludes 0.
6. All results are reported whether better or worse than v1.
