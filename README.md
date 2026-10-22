# Avocado Ripening Stage Classification

Two approaches are compared on the same stratified 70/15/15 split:

1. **Classical ML**: CIELAB + HSV colour histograms/moments + LBP texture, then Random Forest / SVM
2. **Deep learning**: MobileNetV2 transfer learning (frozen early layers, new 128-unit head, early stopping on val loss)

## Setup
```bash
pip install -r requirements.txt
# download the dataset from the Drive link, unzip so that you have data/raw/<class_name>/*.jpg
```

## Run
```bash
python avocado_pipeline.py --data data/raw --epochs 20   # EDA + both models + comparison + latency
python main_demo.py data/raw/<class>/<image>.jpg          # example run with both models
```

Outputs are written to `outputs/` (class distribution plot, audit.csv, confusion matrices,
model_comparison.csv, latency.json, saved models).

## Results
Fill in from `outputs/model_comparison.csv`:

| Model | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 |
|---|---|---|---|---|
| Random Forest | | | | |
| SVM (RBF) | | | | |
| MobileNetV2 | | | | |

## Links
- Colab: <paste link, set to "Anyone with the link can view">
- Report: `report/Assignment2_Avocado_Report.md` (export to PDF)
