# Explainable X-Ray AI: Pleural Effusion Diagnostic Assistant

A deep learning application that detects **Pleural Effusion** from frontal chest X-rays using **EfficientNet-B0 Transfer Learning**. The system combines high-performing medical image classification with **Grad-CAM visual explanations**, enabling users to understand which regions of an X-ray influenced the model's prediction. The model is deployed through an interactive **Streamlit dashboard** for real-time inference and visualization.

---

## 🚀 Key Features

### Transfer Learning with EfficientNet-B0
- Leveraged a pretrained EfficientNet-B0 backbone for medical image classification.
- Implemented a two-stage training strategy using layer freezing and fine-tuning.
- Optimized for improved convergence on specialized chest X-ray data.

### Explainable AI with Grad-CAM
- Integrated Grad-CAM heatmap generation for visual model interpretability.
- Highlights image regions contributing most strongly to the prediction.
- Helps address the black-box nature of deep learning models in healthcare applications.

### Streamlit Deployment
- Built an interactive web application for image upload and inference.
- Generates predictions and Grad-CAM visualizations in real time.
- Designed an in-memory processing pipeline using PIL and NumPy for efficient execution.

### Class Imbalance Handling
- Applied weighted binary classification loss using a custom positive class weight.
- Improved robustness on naturally imbalanced medical datasets.

---

## 📊 Dataset

The model was trained on a filtered subset of the **Stanford CheXpert Dataset**, focusing exclusively on **Frontal Chest X-Rays** to match common clinical workflows.

| Dataset Split | Size |
|--------------|------|
| Training | 114,616 Images |
| Validation | 202 Images |
| Positive Class Ratio | ~40.3% |

The validation dataset consists of expert-labeled studies annotated by Stanford radiologists, providing a high-quality benchmark for evaluation.

---

## 📈 Model Performance

The training pipeline used transfer learning with EfficientNet-B0, Binary Cross-Entropy Loss, and the Adam optimizer.

| Stage | Epoch | Train Loss | Validation Loss | Validation AUC-ROC |
|---------|---------|---------|---------|---------|
| Frozen Backbone | 1 | 0.7367 | 0.5981 | 0.8332 |
| Frozen Backbone | 2 | 0.7244 | 0.6109 | 0.8137 |
| Fine-Tuning | 1 | 0.5999 | 0.4216 | 0.9249 |
| Fine-Tuning | 2 | 0.5496 | 0.3915 | **0.9338** |
| Fine-Tuning | 3 | 0.5287 | 0.3954 | 0.9307 |

### Final Results

✅ Best Validation AUC-ROC: **0.9338**

✅ Transfer Learning + Fine-Tuning Strategy

✅ Explainable Predictions using Grad-CAM

✅ Interactive Streamlit Deployment

The best-performing model checkpoint was automatically saved based on validation AUC-ROC and achieved strong discriminative performance on the expert-annotated validation set.

---

## 🛠️ Project Structure

```text
CNN-GRAD/
├── src/
│   ├── app.py             # Streamlit Dashboard
│   ├── model.py           # EfficientNet-B0 Model Definition
│   ├── gradcam.py         # Grad-CAM Heatmap Generation
│   ├── dataset.py         # Data Processing Pipeline
│   └── best_model.pth     # Best Saved Model Weights
├── requirements.txt
└── README.md
```

---

## 💻 Local Installation

### Clone Repository

```bash
git clone https://github.com/VedantSoni16/CheXpert-EfficientNet-GradCAM.git
cd CheXpert-EfficientNet-GradCAM
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch Application

```bash
streamlit run src/app.py
```
