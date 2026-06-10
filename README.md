# 🔬 Clinical Disease Detection System

> An end-to-end medical image classification system powered by **EfficientNetB4**, with **Grad-CAM++** visual explanations and **Monte Carlo Dropout** Bayesian uncertainty estimation — deployed via a **Streamlit** Clinical Decision Support System (CDSS).

---

## 🚀 Quick Start

### 1. Environment Setup

Clone or enter the directory, create a virtual environment, and install dependencies:

```bash
cd disease_detection_system
python -m venv venv
venv\Scripts\activate          # On Linux/macOS: source venv/bin/activate
pip install -r app/requirements.txt
```

### 2. Dataset Preparation

Your local dataset is automatically structured under `data/raw/isic` containing:
- `data/raw/isic/Train/` (with 9 class directories)
- `data/raw/isic/Test/` (with 9 class directories)

You can run training directly, and the data loaders will automatically scan this directory.

### 3. Model Training

Start the two-phase training loop (Phase 1: Frozen base head-training, Phase 2: Unfrozen top-30 fine-tuning):

```bash
# Run full training on local skin lesion dataset
python -m src.train --mode skin_lesion

# Quick training check using synthetic data (dry-run smoke test)
python -m src.train --dummy --epochs1 2 --epochs2 1
```

Training automatically saves the best recall-optimized weights to `models/checkpoint.weights.h5`.

### 4. Launch the Streamlit CDSS Dashboard

Serve the premium clinical dashboard:

```bash
streamlit run app/streamlit_app.py
```

Open the local URL in your web browser (typically `http://localhost:8501`) to start uploading scans, visualizing heatmaps, evaluating uncertainty, and exporting PDF clinical reports.

### 5. Production Docker Deployment

Containerize the clinical frontend:

```bash
docker build -t disease-detection-system .
docker run -p 8501:8501 disease-detection-system
```

---

## 🧠 Clinical AI Pipeline

```
Upload Image
     │
     ▼
CLAHE Enhancement  ──►  EfficientNetB4 (ImageNet pre-trained base)
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
         Grad-CAM++        Softmax           MC-Dropout
         Heatmap         Prediction        Uncertainty
              │                │                │
              └────────────────┴────────────────┘
                               │
                    Streamlit CDSS Interface
                               │
                          PDF Report
```

### Two-Phase Training Strategy

| Phase | Base Model | Learning Rate | Purpose |
|-------|-----------|---------------|---------|
| **Phase 1** | Frozen | `1e-3` (with cosine decay & warmup) | Warmup and train classification head |
| **Phase 2** | Top 30 layers unfrozen | `1e-5` (low constant) | Fine-tune convolutional filters to dermatoscopic details |

### Key Clinical-Safety Design Decisions
- **Recall-optimized Checkpoints**: Model saving and early stopping track **Recall** to minimize false negatives (critical in medical diagnostics, where missing a pathology is far worse than a false positive).
- **Epistemic Uncertainty Filter**: We run 30 stochastic forward passes with dropout enabled at inference time. The mean is used as the prediction, and the entropy/standard deviation determines the confidence level. High uncertainty automatically routes the scan to urgent specialist triage.
- **CLAHE Enhancement**: Standardizes image exposure to highlight dermatoscopic boundaries and colors.
- **Label Smoothing (ε=0.1)**: Calibrates probabilities, preventing the network from becoming overconfident.

---

## 📁 Project Structure

```
disease_detection_system/
├── app/
│   ├── streamlit_app.py        # Streamlit clinical CDSS frontend & PDF exporter
│   └── requirements.txt        # App-specific package dependencies
├── data/
│   ├── raw/                    # Original, unaltered images (ISIC)
│   └── processed/              # Preprocessed splits
├── models/
│   ├── checkpoint.weights.h5   # Trained model weights file
│   └── model_card.md           # Model transparency documentation
├── notebooks/
│   └── 01_eda_and_baseline.ipynb # EDA and baseline verification notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Preprocessors, class-weight, tf.data loader
│   ├── model.py                # EfficientNetB4 + Grad-CAM++ logic
│   ├── train.py                # Two-phase training script
│   └── utils.py                # Plotting helpers, MC-Dropout, and clinical config
├── Dockerfile                  # Production container configuration
└── README.md                   # Setup guide and technical overview
```

---

## 📄 License
This CDSS framework is distributed under the MIT License.
