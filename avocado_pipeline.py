"""
Avocado ripening-stage classification: two approaches compared.
  Approach 1: CIELAB + HSV colour features + LBP texture -> Random Forest / SVM
  Approach 2: MobileNetV2 transfer learning (frozen early layers, new head, early stopping)

Dataset layout after unzipping the Drive file:   data/raw/<class_name>/*.jpg
Run:   python avocado_pipeline.py --data data/raw --epochs 20
Outputs (in ./outputs): audit.csv, plots, confusion matrices, model_comparison.csv,
                        classical_best.joblib, mobilenet_best.pt, latency.json
"""
import argparse, hashlib, json, random, time
from pathlib import Path

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
random.seed(SEED); np.random.seed(SEED)
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
IMNET_MEAN, IMNET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# =============================================================== EDA / audit
SPLIT_NAMES = {"train", "test", "val", "valid", "validation"}


def load_paths(data_dir):
    """Works for data/raw/<class>/img.jpg AND data/raw/train|test|val/<class>/img.jpg.
    Label = name of the folder directly containing the image. All splits are pooled,
    then re-split 70/15/15 in main() (exact duplicates are dropped first to avoid leakage)."""
    rows = []
    for p in sorted(Path(data_dir).rglob("*")):
        if p.is_file() and p.suffix.lower() in EXTS:
            label = p.parent.name
            if label.lower() in SPLIT_NAMES:      # image sits directly in train/test: no class info
                continue
            rows.append((str(p), label))
    df = pd.DataFrame(rows, columns=["path", "label"])
    if df.empty:
        raise SystemExit(f"\nNo images found under '{data_dir}'.\n"
                         "Put the dataset class folders (or train/test folders) inside that folder, "
                         "then run again. Check with:  dir data\\raw")
    print("Found classes:", dict(df.label.value_counts()))
    return df


def audit(df):
    recs = []
    for p in df.path:
        try:
            img = cv2.imread(p)
            if img is None:
                raise ValueError
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            glare = float(((hsv[..., 2] > 245) & (hsv[..., 1] < 40)).mean())   # specular highlights
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            recs.append(dict(path=p, w=w, h=h, brightness=gray.mean(),
                             blur=cv2.Laplacian(gray, cv2.CV_64F).var(), glare=glare,
                             mean_a=lab[..., 1].mean(),
                             md5=hashlib.md5(open(p, "rb").read()).hexdigest(), ok=True))
        except Exception:
            recs.append(dict(path=p, ok=False))
    meta = df.merge(pd.DataFrame(recs), on="path")
    counts = meta.label.value_counts()
    print("\n=== CLASS DISTRIBUTION ===\n", counts)
    print(f"Imbalance ratio (max/min): {counts.max() / counts.min():.2f}")
    print("Corrupt files:", int((~meta.ok).sum()))
    good = meta[meta.ok].copy()
    print("Exact duplicate files:", int(good.duplicated("md5", keep=False).sum()))
    good["area"] = good.w * good.h
    for col in ["brightness", "blur", "glare", "area"]:
        q1, q3 = good[col].quantile([.25, .75]); iqr = q3 - q1
        n = int(((good[col] < q1 - 1.5 * iqr) | (good[col] > q3 + 1.5 * iqr)).sum())
        print(f"IQR outliers in {col}: {n}")
    print("\nPer-class means (capture-bias check):\n",
          good.groupby("label")[["brightness", "blur", "glare", "mean_a", "w", "h"]].mean().round(2))
    counts.plot.bar(color="forestgreen", title="Avocado ripening stage distribution")
    plt.tight_layout(); plt.savefig(OUT / "class_distribution.png"); plt.close()
    good.boxplot(column="mean_a", by="label"); plt.suptitle("")
    plt.title("CIELAB a* (green->red) by class"); plt.savefig(OUT / "a_channel_by_class.png"); plt.close()
    good.to_csv(OUT / "audit.csv", index=False)
    return good


# ===================================================== Approach 1: classical
def extract_features(path, size=128):
    img = cv2.resize(cv2.imread(path), (size, size))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    f = []
    for im, spec in [(hsv, [(0, 24, 180), (1, 16, 256), (2, 16, 256)]),
                     (lab, [(0, 16, 256), (1, 16, 256), (2, 16, 256)])]:   # a* = green->red axis
        for ch, bins, rng in spec:
            h = cv2.calcHist([im], [ch], None, [bins], [0, rng]).flatten()
            f.extend(h / (h.sum() + 1e-8))
    for im in (hsv, lab):                                  # colour moments
        for c in range(3):
            f.extend([im[..., c].mean(), im[..., c].std()])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)           # LBP texture: smooth vs pebbled skin
    lbp = local_binary_pattern(gray, 8, 1, "uniform")
    h, _ = np.histogram(lbp, bins=10, range=(0, 10))
    f.extend(h / (h.sum() + 1e-8))
    return np.array(f, dtype=np.float32)


def run_classical(tr, va, te):
    X = lambda d: np.stack([extract_features(p) for p in d.path])
    Xtr, Xva, Xte = X(tr), X(va), X(te)
    sc = StandardScaler().fit(Xtr)
    cands = {
        "RandomForest": RandomForestClassifier(300, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "SVM_RBF": SVC(C=10, kernel="rbf", class_weight="balanced", probability=True, random_state=SEED),
    }
    results, best_name, best_f1 = {}, None, -1
    for name, m in cands.items():
        m.fit(sc.transform(Xtr), tr.y.values)
        vf1 = f1_score(va.y.values, m.predict(sc.transform(Xva)), average="macro")
        print(f"{name}: val macro-F1 {vf1:.4f}")
        results[name] = (te.y.values, m.predict(sc.transform(Xte)))
        if vf1 > best_f1:
            best_name, best_f1 = name, vf1
    return results, best_name, cands[best_name], sc


# ===================================================== Approach 2: MobileNetV2
def build_model(num_classes, pretrained=True):
    import torch.nn as nn
    from torchvision import models
    w = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.mobilenet_v2(weights=w)
    for p in net.features[:14].parameters():               # freeze early feature extractors
        p.requires_grad = False
    net.classifier[1] = nn.Sequential(nn.Linear(net.last_channel, 128), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(128, num_classes))
    return net


def eval_transform():
    from torchvision import transforms as T
    return T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(IMNET_MEAN, IMNET_STD)])


def run_cnn(tr, va, te, classes, epochs, patience=4, bs=32):
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms as T

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    norm = T.Normalize(IMNET_MEAN, IMNET_STD)
    train_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.7, 1)), T.RandomHorizontalFlip(),
                          T.RandomRotation(20), T.ColorJitter(0.2, 0.2, 0.0, 0.0),   # brightness/contrast only:
                          T.ToTensor(), norm])                                      # hue/saturation ARE the label signal

    class DS(Dataset):
        def __init__(s, d, tf): s.p, s.y, s.tf = d.path.values, d.y.values, tf
        def __len__(s): return len(s.p)
        def __getitem__(s, i): return s.tf(Image.open(s.p[i]).convert("RGB")), int(s.y[i])

    mk = lambda d, tf, sh: DataLoader(DS(d, tf), batch_size=bs, shuffle=sh, num_workers=0)
    tr_l, va_l, te_l = mk(tr, train_tf, True), mk(va, eval_transform(), False), mk(te, eval_transform(), False)

    net = build_model(len(classes)).to(dev)
    cnt = np.bincount(tr.y.values, minlength=len(classes))
    w = torch.tensor(cnt.sum() / (len(classes) * cnt), dtype=torch.float32).to(dev)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    def evaluate(loader):
        net.eval(); ys, ps, tot = [], [], 0.0
        with torch.no_grad():
            for x, y in loader:
                o = net(x.to(dev)); tot += loss_fn(o, y.to(dev)).item() * len(y)
                ps += o.argmax(1).cpu().tolist(); ys += y.tolist()
        return np.array(ys), np.array(ps), tot / len(ys)

    best, bad = 1e9, 0
    for ep in range(epochs):
        net.train()
        for x, y in tr_l:
            opt.zero_grad(); loss_fn(net(x.to(dev)), y.to(dev)).backward(); opt.step()
        sched.step()
        yv, pv, vl = evaluate(va_l)
        print(f"epoch {ep+1}/{epochs}  val loss {vl:.4f}  val macro-F1 {f1_score(yv, pv, average='macro'):.4f}")
        if vl < best:
            best, bad = vl, 0
            torch.save({"state": net.state_dict(), "classes": classes}, OUT / "mobilenet_best.pt")
        else:
            bad += 1
            if bad >= patience:
                print("early stopping"); break
    net.load_state_dict(torch.load(OUT / "mobilenet_best.pt")["state"])
    y, p, _ = evaluate(te_l)
    return y, p, net.cpu().eval()


# =============================================================== reporting
def report(name, y, p, classes, rows):
    print(f"\n===== {name} =====")
    print(classification_report(y, p, target_names=classes, digits=3))
    rows.append(dict(model=name, accuracy=accuracy_score(y, p),
                     macro_precision=precision_score(y, p, average="macro", zero_division=0),
                     macro_recall=recall_score(y, p, average="macro", zero_division=0),
                     macro_f1=f1_score(y, p, average="macro"),
                     weighted_f1=f1_score(y, p, average="weighted")))
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y, p), annot=True, fmt="d", cmap="Greens",
                xticklabels=classes, yticklabels=classes)
    plt.title(name); plt.ylabel("True"); plt.xlabel("Predicted"); plt.tight_layout()
    plt.savefig(OUT / f"cm_{name}.png"); plt.close()


def benchmark(paths, clf, scaler, net, n=30):
    """Measured CPU latency per image (ms): answers 'how does it perform in production'."""
    import torch
    tf, paths = eval_transform(), list(paths)[:n]
    t = time.perf_counter()
    for p in paths:
        clf.predict(scaler.transform(extract_features(p)[None]))
    classical = (time.perf_counter() - t) / len(paths) * 1000
    t = time.perf_counter()
    with torch.no_grad():
        for p in paths:
            net(tf(Image.open(p).convert("RGB"))[None])
    cnn = (time.perf_counter() - t) / len(paths) * 1000
    res = {"classical_ms_per_image_cpu": round(classical, 1), "mobilenetv2_ms_per_image_cpu": round(cnn, 1),
           "mobilenetv2_params_M": round(sum(p.numel() for p in net.parameters()) / 1e6, 2)}
    print("\nLatency:", res)
    json.dump(res, open(OUT / "latency.json", "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--epochs", type=int, default=20)
    a = ap.parse_args()

    df = audit(load_paths(a.data))
    classes = sorted(df.label.unique())
    df["y"] = df.label.map({c: i for i, c in enumerate(classes)})
    df = df.drop_duplicates("md5")                          # avoid train/test leakage
    trva, te = train_test_split(df, test_size=0.15, stratify=df.y, random_state=SEED)
    tr, va = train_test_split(trva, test_size=0.15 / 0.85, stratify=trva.y, random_state=SEED)
    print(f"\nSplit -> train {len(tr)} / val {len(va)} / test {len(te)}")

    rows = []
    res, best_name, clf, sc = run_classical(tr, va, te)
    for n, (y, p) in res.items():
        report(n, y, p, classes, rows)
    joblib.dump({"model": clf, "scaler": sc, "classes": classes, "name": best_name}, OUT / "classical_best.joblib")

    y, p, net = run_cnn(tr, va, te, classes, a.epochs)
    report("MobileNetV2", y, p, classes, rows)

    summary = pd.DataFrame(rows).round(4)
    print("\n", summary.to_string(index=False))
    summary.to_csv(OUT / "model_comparison.csv", index=False)
    benchmark(te.path.values, clf, sc, net)


if __name__ == "__main__":
    main()