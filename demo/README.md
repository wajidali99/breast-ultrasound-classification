---
title: Breast Ultrasound Classifier
emoji: 🩺
colorFrom: indigo
colorTo: pink
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Research demo — benign vs malignant breast ultrasound
---

# Breast Ultrasound Lesion Classifier (research demo)

⚠️ **Not a medical device. Research use only.**

Five DenseNet121 models (v2b: mask-guided training + aspect-ratio augmentation) trained on BUSI vote benign vs malignant; Grad-CAM shows the supporting regions.

- Model weights: [Wajiddev99/busi-densenet121-v2b](https://huggingface.co/Wajiddev99/busi-densenet121-v2b)
- Code and evaluation: [GitHub](https://github.com/wajidali99/breast-ultrasound-classification)
