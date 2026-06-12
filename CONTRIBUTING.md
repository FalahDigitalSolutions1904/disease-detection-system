# Disease Detection System — Contribution Guide

Thank you for your interest in contributing to this clinical AI project!

## Setting Up the Dev Environment

```bash
git clone https://github.com/FalahDigitalSolutions1904/disease-detection-system.git
cd disease-detection-system
python -m venv venv
venv\Scripts\activate
pip install -r app/requirements.txt
```

## Running Tests

```bash
# Syntax + import checks
python -m py_compile src/train.py src/model.py src/utils.py src/data_loader.py
python -m py_compile src/evaluate.py src/preprocess_data.py

# Dummy training run (fast, no data needed)
python -m src.train --dummy --epochs1 1 --epochs2 1

# Dummy evaluation run
python -m src.evaluate --dummy
```

## Code Style

- Follow [PEP 8](https://pep8.org/)
- Add docstrings to all public functions
- Write type hints for all function arguments and return values

## Contribution Areas

| Area | Description |
|------|-------------|
| 🧠 Model | Improving EfficientNetB4 head or adding new architectures |
| 🔬 Explainability | Extending Grad-CAM++ with LIME or SHAP |
| 📊 Evaluation | Adding new clinical metrics (AUROC, sensitivity @ specificity) |
| 🖥️ UI | Improving the Streamlit CDSS dashboard |
| 🐳 DevOps | Docker / CI-CD improvements |
| 📄 Docs | Documentation, tutorials, or model card updates |

## Branch Strategy

- `master` — stable, production-ready code  
- `feat/<feature-name>` — new features  
- `fix/<issue-name>` — bug fixes  
- `docs/<topic>` — documentation updates  

## Pull Request Checklist

- [ ] Code passes syntax checks
- [ ] Docstrings are added / updated
- [ ] Dummy training run passes without errors
- [ ] README is updated if public API changed
