"""Single-run demo: classify one avocado image with BOTH approaches.
Usage: python main_demo.py path/to/avocado.jpg
Requires outputs/classical_best.joblib and outputs/mobilenet_best.pt (created by avocado_pipeline.py).
"""
import sys
from pathlib import Path

import joblib
import torch
from PIL import Image

from avocado_pipeline import OUT, build_model, eval_transform, extract_features


def run_avocado_demo(image_path):
    # Approach 1: classical colour + texture features
    bundle = joblib.load(OUT / "classical_best.joblib")
    x = bundle["scaler"].transform(extract_features(image_path)[None])
    probs = bundle["model"].predict_proba(x)[0]
    c_idx = int(probs.argmax())

    # Approach 2: MobileNetV2 (same ImageNet normalisation as in training)
    ck = torch.load(OUT / "mobilenet_best.pt", map_location="cpu")
    net = build_model(len(ck["classes"]), pretrained=False)
    net.load_state_dict(ck["state"]); net.eval()
    with torch.no_grad():
        dl_probs = torch.softmax(net(eval_transform()(Image.open(image_path).convert("RGB"))[None]), 1)[0]
    d_idx = int(dl_probs.argmax())

    print(f"--- Prediction results for {image_path} ---")
    print(f"Approach 1 ({bundle['name']}): {bundle['classes'][c_idx]}  (confidence {probs[c_idx]:.2f})")
    print(f"Approach 2 (MobileNetV2):  {ck['classes'][d_idx]}  (confidence {dl_probs[d_idx]:.2f})")
    print("MobileNetV2 class probabilities:",
          {c: round(float(p), 3) for c, p in zip(ck["classes"], dl_probs)})


def pick_random_test_image(root="data/raw/test"):
    import random
    imgs = [p for p in Path(root).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    if not imgs:
        raise SystemExit(f"No images found under {root}")
    p = random.choice(imgs)
    print(f"(no image given: picked a random one. True class folder: '{p.parent.name}')")
    return str(p)


if __name__ == "__main__":
    # Usage: python main_demo.py                -> random test image
    #        python main_demo.py path/to/img.jpg -> that image
    path = sys.argv[1] if len(sys.argv) > 1 else pick_random_test_image()
    if not Path(path).is_file():
        raise SystemExit(f"File not found: {path}\nTip: run  python main_demo.py  with no argument to use a random test image.")
    run_avocado_demo(path)