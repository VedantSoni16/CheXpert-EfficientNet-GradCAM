# Explainable X-Ray AI: Pleural Effusion Diagnostic Assistant

A production-ready, deep learning application that identifies **Pleural Effusion** from frontal chest X-rays using an optimized **EfficientNet-B0** architecture. The system features an in-memory **Grad-CAM (Gradient-weighted Class Activation Mapping)** layer to provide clinically explainable visual heatmaps, deployed as an interactive web dashboard via Streamlit.

---

## 🚀 Key Features
- **Two-Stage Training Pipeline:** Implemented freezing/unfreezing schedules to maximize transfer learning efficiency on specialized medical imagery.
- **Visual Interpretability:** Integrated Grad-CAM to highlight localized pathological features, tackling the deep learning "black box" dilemma in healthcare.
- **In-Memory Optimization:** Reconstructed the web app to process inputs entirely via memory arrays (`PIL` / `NumPy`), preventing server-side I/O bottlenecks.
- **Imbalance Mitigation:** Calculated and applied custom loss penalties (`pos_weight = 1.48`) to compensate for medical class distributions.

---

## 📊 Dataset & Performance

The model was trained on a highly filtered subset of the **Stanford CheXpert dataset**, focusing exclusively on **Frontal views** to align with standard clinical diagnostic workflows.

- **Training Set:** 114,616 Images (Stratified Class Distribution: ~40.3% Positive)
- **Validation Set:** 202 Images (Gold-standard panel, manually annotated by 3 board-certified Stanford Radiologists)

### Training History & Validation Results
The training sequence utilized binary cross-entropy with logits paired with an Adam optimizer ($LR = 10^{-4}$ during fine-tuning). 

| Stage | Epoch | Train Loss | Val Loss | Val AUC-ROC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Stage 1 (Frozen Backbone)** | 1 | 0.7367 | 0.5981 | 0.8332 | Initialized |
| **Stage 1 (Frozen Backbone)** | 2 | 0.7244 | 0.6109 | 0.8137 | Stagnant |
| **Stage 2 (Fine-Tuning)** | 1 | 0.5999 | 0.4216 | 0.9249 | Checkpointed |
| **Stage 2 (Fine-Tuning)** | 2 | 0.5496 | 0.3915 | **0.9338** | **Best Weights Saved** |
| **Stage 2 (Fine-Tuning)** | 3 | 0.5287 | 0.3954 | 0.9307 | Early Overfitting Sign |

*The checkpoint configuration automatically isolated and preserved the peak state at Epoch 2 ($0.9338$ AUC-ROC).*

---

## 🛠️ Project Structure
```text
CNN-GRAD/
├── src/
│   ├── app.py             # Streamlit Interactive Dashboard
│   ├── model.py           # EfficientNet-B0 Architecture Definitions
│   ├── gradcam.py         # Gradient Hook Extractor & Map Overlays
│   ├── dataset.py         # CheXpert Parsing and Augmentation Pipeline
│   └── best_model.pth     # Serialized Model Weights (Peak Validation State)
├── requirements.txt       # Production Dependencies
└── README.md
💻 Local Installation & SetupClone the Repository:Bashgit clone [https://github.com/YOUR_GITHUB_USERNAME/Explainable-XRay-AI.git](https://github.com/YOUR_GITHUB_USERNAME/Explainable-XRay-AI.git)
cd Explainable-XRay-AI
Install Dependencies:Bashpip install -r requirements.txt
Run the Dashboard:Bashpython -m streamlit run src/app.py
🔬 Mathematical Breakdown: Grad-CAM ExplainabilityThe explainability layer targets the final convolutional feature maps $A^k$ of the EfficientNet backbone. Gradients of the single output logit score $Y$ with respect to the feature maps are globally pooled across spatial dimensions $(U \times V)$ to compute importance weights $\alpha_k$:$$\alpha_k = \frac{1}{U \times V} \sum_{i=1}^{U} \sum_{j=1}^{V} \frac{\partial Y}{\partial A_{i,j}^k}$$A weighted forward linear combination followed by a Rectified Linear Unit ($\text{ReLU}$) ensures the heatmap only registers features that positively correlate with the target pathology:$$L_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_{k} \alpha_k A^k\right)$$