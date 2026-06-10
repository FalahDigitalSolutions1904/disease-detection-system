# Model Card — EfficientNetB4 Disease Detector

This model card details the design, characteristics, and limitations of the clinical decision support model. It serves as a transparent record of the model's capabilities to promote responsible clinical use.

## Model Details
- **Developer**: Falah Digital Solutions / ML Intern
- **Architecture**: EfficientNetB4 with custom classification head (GAP -> BN -> Dropout -> Dense -> Dropout -> Dense Softmax)
- **Framework**: TensorFlow 2.13.0 / Keras 2.13.1
- **Input Dimensions**: 380 × 380 × 3 (RGB)
- **Output Classes**: Multi-class softmax probabilities (classes depend on selected modality)
- **Uncertainty Method**: Monte Carlo (MC) Dropout (N=30 stochastic forward passes at inference time)
- **Interpretability Method**: Grad-CAM++ (attentional gradient maps overlay)

## Intended Use
- **Primary Use Case**: Clinical Decision Support System (CDSS) to assist clinicians (radiologists, dermatologists, etc.) in early screening of pathological findings.
- **Out of Scope**: Autonomous decision-making. The model is NOT intended to serve as a stand-alone diagnostic tool. It must always be reviewed by a licensed clinician.
- **Target Population**: Frontal chest radiographs, retinal fundus photographs, and dermatoscopic skin lesions.

## Clinical Guidelines & Risk Strategy
- **False Negative Mitigation**: In clinical settings, false negatives (missing a disease) carry far higher risk than false positives. Thus, the model checkpointing and early stopping are explicitly optimized for **Recall** rather than raw Accuracy.
- **Bayesian Safety Net**: The MC-Dropout uncertainty estimation serves as a safety filter. If the model generates a high-uncertainty label, the case is automatically flagged for urgent specialist triage regardless of the predicted class.

## Quantitative Evaluation Results

*(This section is updated upon completion of training and evaluation runs)*

| Modality / Class | Precision | Recall | F1-Score |
| --- | --- | --- | --- |
| **Chest X-Ray** | | | |
| - Normal | TBD | TBD | TBD |
| - Pneumonia | TBD | TBD | TBD |
| **Retinal Scan** | | | |
| - 5 severity grades | TBD | TBD | TBD |
| **Skin Lesion** | | | |
| - 9 diagnostic classes | TBD | TBD | TBD |

## Training Data Details
The pipeline is designed to ingest and train on the following gold-standard medical benchmarks:
1. **Chest X-Ray Modality**: NIH Chest X-Ray Dataset (5,863 images) targeting pathologies like Pneumonia.
2. **Retinal Scan Modality**: APTOS 2019 Blindness Detection (3,662 high-resolution fundus photographs) graded on the 5-point diabetic retinopathy scale.
3. **Skin Lesion Modality**: ISIC Skin Lesion archive dataset (dermatoscopic images) classifying 9 diagnostic categories (melanoma, nevus, basal cell carcinoma, etc.).

## Preprocessing & Pipeline Integrity
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Contrast enhancement is applied to the L-channel of the LAB color space to reveal subtle pathological variations.
- **Conservative Augmentation**: Horizontal/vertical flips, minor rotations (15°), and light zooms. Distortions are restricted to preserve anatomical validity.
- **Class-Imbalance Handling**: Computed class weights are applied via the sklearn balanced formula to prevent the model from biasing toward the majority class.

## Bias & Limitations
- **Demographic Bias**: Performance on skin lesions is known to vary significantly across different Fitzpatrick skin types. High error rates may occur on darker skin tones if not sufficiently represented in training.
- **Acquisition Artifacts**: Variations in scanner manufacturer, exposure, and image compression can lead to domain shifts and increased uncertainty.
- **Clinical Validation**: This model has not undergone clinical trials and holds no FDA 510(k) clearance or EMA approval.

## Ethical Considerations
- **Human-in-the-Loop**: Diagnostics must remain user-controlled. Clinicians must be able to override recommendations.
- **Data Privacy**: Medical scans contain protected health information (PHI). Implement proper HIPAA/GDPR safeguards when deploying this system in clinical environments.

## Version History
- **v1.0.0 (Current)**: Initial release with EfficientNetB4 transfer learning, Grad-CAM++, MC-dropout uncertainty, and premium clinical interface.
