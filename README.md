# Explainable X-Ray AI: Pleural Effusion Diagnostic Assistant

[![Live Demo](https://img.shields.io/badge/Streamlit-Live_Demo-red)](https://chexpert-efficientnet-gradcam-ffca7n8u9a5uxxafze6squ.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/VedantSoni16/CheXpert-EfficientNet-GradCAM)

A deep learning application that detects **Pleural Effusion** from frontal chest X-rays using **EfficientNet-B0 Transfer Learning**. The system combines high-performing medical image classification with **Grad-CAM visual explanations**, enabling users to understand which regions of an X-ray influenced the model's prediction. The model is deployed through an interactive **Streamlit dashboard** for real-time inference and visualization.

---

## 🚀 Key Features

### Transfer Learning with EfficientNet-B0

* Leveraged a pretrained EfficientNet-B0 backbone for medical image classification.
* Implemented a two-stage training strategy using layer freezing and fine-tuning.
* Achieved strong performance on a large-scale medical imaging dataset.

### Explainable AI with Grad-CAM

* Integrated Grad-CAM heatmap generation for visual model interpretability.
* Highlights image regions contributing most strongly to model predictions.
* Improves transparency of deep learning decisions in healthcare applications.

### Streamlit Deployment

* Built an interactive web application for image upload and inference.
* Generates predictions and Grad-CAM visualizations in real time.
* Uses an optimized in-memory processing pipeline with PIL and NumPy.

### Class Imbalance Handling

* Applied weighted binary cross-entropy loss using custom class weights.
* Improved robustness on imbalanced medical imaging data.

---

## 📊 Dataset

The model was trained on a filtered subset of the Stanford CheXpert Dataset, focusing exclusively on frontal chest X-rays.

| Dataset Split        | Size           |
| -------------------- | -------------- |
| Training             | 114,616 Images |
| Validation           | 202 Images     |
| Positive Class Ratio | ~40.3%         |

---

## 📈 Model Performance

| Stage           | Epoch | Train Loss | Validation Loss | Validation AUC-ROC |
| --------------- | ----- | ---------- | --------------- | ------------------ |
| Frozen Backbone | 1     | 0.7367     | 0.5981          | 0.8332             |
| Frozen Backbone | 2     | 0.7244     | 0.6109          | 0.8137             |
| Fine-Tuning     | 1     | 0.5999     | 0.4216          | 0.9249             |
| Fine-Tuning     | 2     | 0.5496     | 0.3915          | **0.9338**         |
| Fine-Tuning     | 3     | 0.5287     | 0.3954          | 0.9307             |

### Final Results

✅ Best Validation AUC-ROC: **0.9338**

✅ EfficientNet-B0 Transfer Learning

✅ Grad-CAM Explainability

✅ Interactive Streamlit Deployment

---

## 🛠️ Project Structure

```text
CNN-GRAD/
├── src/
│   ├── app.py
│   ├── model.py
│   ├── gradcam.py
│   ├── dataset.py
│   └── best_model.pth
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

---

## 🌐 Live Demo

Try the deployed application here:

**https://chexpert-efficientnet-gradcam-ffca7n8u9a5uxxafze6squ.streamlit.app/**

---

## 🎯 Example Workflow

1. Upload a frontal chest X-ray image.
2. The EfficientNet-B0 model generates a Pleural Effusion probability score.
3. Grad-CAM identifies the regions influencing the prediction.
4. The dashboard displays:

   * Prediction Confidence
   * Classification Result
   * Explainability Heatmap Overlay

---


