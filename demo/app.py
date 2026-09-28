"""Breast ultrasound classifier demo — research only, not for diagnosis.
Five DenseNet121 fold models (v2b) vote benign/malignant; Grad-CAM shows the regions that support the result.
"""
import json

import gradio as gr
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from matplotlib import colormaps
from PIL import Image
from torchvision.transforms import functional as TF

try:                                   # ZeroGPU hardware requires a @spaces.GPU function;
    import spaces                      # on CPU hardware this decorator does nothing
    gpu_decorator = spaces.GPU
except ImportError:
    gpu_decorator = lambda f: f

MODEL_REPO = "Wajiddev99/busi-densenet121-v2b"
SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
torch.set_num_threads(2)


# ---------------------------------------------------------------- model (same as src/v2.py)
class DenseNetSeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("densenet121", pretrained=False, num_classes=2, drop_rate=0.2)
        self.seg_head = nn.Conv2d(self.backbone.num_features, 1, kernel_size=1)

    def forward(self, x):
        return self.backbone.forward_head(self.backbone.forward_features(x))


cfg = json.load(open(hf_hub_download(MODEL_REPO, "config.json")))
THR = float(cfg["threshold_per_model"])
MODELS = []
for k in range(cfg["n_models"]):
    m = DenseNetSeg()
    m.load_state_dict(torch.load(hf_hub_download(MODEL_REPO, f"v2b_fold{k}.pt"), map_location="cpu"))
    MODELS.append(m.eval())


# ---------------------------------------------------------------- preprocessing (same as training)
def letterbox(img, size=SIZE):
    w, h = img.size
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    canvas = Image.new(img.mode, (size, size), 0)
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), ((size - nw) // 2, (size - nh) // 2))
    content = np.zeros((size, size), bool)
    content[(size - nh) // 2:(size - nh) // 2 + nh, (size - nw) // 2:(size - nw) // 2 + nw] = True
    return canvas, content


def gradcam(model, x, cls):
    store = {}

    def hook(module, inp, out):                       # must return None (a return value would replace the output)
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    h = model.backbone.features.register_forward_hook(hook)
    with torch.enable_grad():
        model.zero_grad(set_to_none=True)
        logits = model(x)
        logits[0, cls].backward()
    h.remove()
    w = store["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * store["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0, 0]
    cam = cam / cam.max().clamp_min(1e-8)
    return torch.softmax(logits.detach(), 1)[0, 1].item(), cam.detach().numpy()


@gpu_decorator
def predict(image):
    if image is None:
        return {}, None, "Please upload an ultrasound image."
    gray = image.convert("L")
    lb, content = letterbox(gray)
    x = TF.normalize(TF.to_tensor(lb.convert("RGB")), MEAN, STD)[None]

    with torch.no_grad():
        probs = [torch.softmax(m(x), 1)[0, 1].item() for m in MODELS]
    votes = sum(p >= THR for p in probs)
    label = "malignant" if votes >= 3 else "benign"
    cls = 1 if label == "malignant" else 0
    cams = [gradcam(m, x, cls)[1] for m in MODELS]
    cam = np.mean(cams, 0); cam = np.where(content, cam / max(cam.max(), 1e-8), 0)

    base = np.array(lb.convert("RGB")).astype(np.float32) / 255
    heat = colormaps["jet"](cam)[..., :3]
    alpha = 0.4 * content[..., None]
    overlay = Image.fromarray((255 * (base * (1 - alpha) + heat * alpha)).clip(0, 255).astype(np.uint8))

    mean_p = float(np.mean(probs))
    details = (f"**Model votes:** {votes} of 5 models say *malignant* (each model's cut-off: P ≥ {THR:.2f})  \n"
               f"**Average P(malignant):** {mean_p:.2f}  \n"
               f"**Individual models:** " + ", ".join(f"{p:.2f}" for p in probs) + "\n\n"
               "Heatmap: red = regions that most support the predicted class (Grad-CAM, averaged over 5 models).")
    return {"malignant": votes / 5, "benign": 1 - votes / 5}, overlay, details


DESCRIPTION = """
### ⚠️ Research demo only — NOT for medical diagnosis
Upload a **breast ultrasound image that contains a lesion**. Five DenseNet121 models trained on the BUSI dataset vote *benign* or *malignant*,
and a Grad-CAM heatmap shows where they looked.

**Known limitation:** on an independent hospital dataset (BUS-BRA, Brazil) accuracy dropped substantially (AUC 0.97 → 0.78).
Results on images from other scanners or hospitals may be unreliable.
Code and full evaluation: [GitHub](https://github.com/wajidali99/breast-ultrasound-classification)
"""

with gr.Blocks(title="Breast Ultrasound Classifier (research demo)") as demo:
    gr.Markdown("# Breast Ultrasound Lesion Classifier")
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Ultrasound image")
            btn = gr.Button("Analyse", variant="primary")
        with gr.Column():
            out_label = gr.Label(label="Share of model votes")
            out_img = gr.Image(label="Grad-CAM heatmap")
            out_md = gr.Markdown()
    btn.click(predict, inputs=inp, outputs=[out_label, out_img, out_md])

if __name__ == "__main__":
    demo.launch()
