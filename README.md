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

The model was trained on a highly filtered subset of the **Stanford CheXpert dataset**, focusing exclusively on **Frontal Views** to align with standard clinical diagnostic workflows.

- **Training Set:** 114,616 Images (~40.3% Positive Class)
- **Validation Set:** 202 Images (Gold-standard panel annotated by 3 board-certified Stanford radiologists)

### Training History & Validation Results

The training sequence utilized Binary Cross-Entropy with Logits Loss paired with the Adam optimizer (`LR = 1e-4`) during fine-tuning.

| Stage | Epoch | Train Loss | Val Loss | Val AUC-ROC | Status |
|---------|---------|---------|---------|---------|---------|
| **Stage 1 (Frozen Backbone)** | 1 | 0.7367 | 0.5981 | 0.8332 | Initialized |
| **Stage 1 (Frozen Backbone)** | 2 | 0.7244 | 0.6109 | 0.8137 | Stagnant |
| **Stage 2 (Fine-Tuning)** | 1 | 0.5999 | 0.4216 | 0.9249 | Checkpointed |
| **Stage 2 (Fine-Tuning)** | 2 | 0.5496 | 0.3915 | **0.9338** | **Best Weights Saved** |
| **Stage 2 (Fine-Tuning)** | 3 | 0.5287 | 0.3954 | 0.9307 | Early Overfitting Sign |

*The checkpoint configuration automatically preserved the highest-performing model from Epoch 2 (Validation AUC-ROC = 0.9338).*

---

## 🛠️ Project Structure

```text
CNN-GRAD/
├── src/
│   ├── app.py             # Streamlit Interactive Dashboard
│   ├── model.py           # EfficientNet-B0 Architecture Definitions
│   ├── gradcam.py         # Gradient Hook Extractor & Heatmap Generation
│   ├── dataset.py         # CheXpert Parsing and Augmentation Pipeline
│   └── best_model.pth     # Best Saved Model Weights
├── requirements.txt
└── README.md
💻 Local Installation & Setup
1. Clone the Repository
git clone https://github.com/VedantSoni16/CheXpert-EfficientNet-GradCAM.git
cd CheXpert-EfficientNet-GradCAM
2. Install Dependencies
pip install -r requirements.txt
3. Run the Dashboard
streamlit run src/app.py
🔬 Mathematical Breakdown: Grad-CAM Explainability

The explainability layer targets the final convolutional feature maps A
k
 produced by the EfficientNet backbone.

Gradients of the output logit Y with respect to each feature map are globally averaged across the spatial dimensions (U×V) to compute feature importance weights:

α
k
	​

=
U×V
1
	​

i=1
∑
U
	​

j=1
∑
V
	​

∂A
i,j
k
	​

∂Y
	​


The Grad-CAM localization map is then generated as:

L
Grad−CAM
	​

=ReLU(
k
∑
	​

α
k
	​

A
k
)

The ReLU operation removes negative contributions, ensuring that only features positively associated with Pleural Effusion are visualized.
