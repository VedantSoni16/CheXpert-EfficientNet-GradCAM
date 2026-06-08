import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Import your custom modules
from model import ChestXrayModel
from gradcam import GradCAM, overlay_heatmap_from_array

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="X-Ray AI Assistant", page_icon="🫁", layout="wide")

st.title("🫁 Pleural Effusion Diagnostic Assistant")
st.markdown("""
Upload a standard frontal chest X-ray. The EfficientNet-B0 model will analyze the image for **Pleural Effusion** and generate a **Grad-CAM heatmap** to explain which visual features led to its diagnosis.
""")

# ==========================================
# MODEL LOADING (Cached)
# ==========================================
@st.cache_resource
def load_model_and_cam():
    # Load architecture
    model = ChestXrayModel(freeze_backbone=False)
    
    # Load weights
    try:
        model.load_state_dict(torch.load("src/best_model.pth", map_location=torch.device('cpu')))
    except Exception as e:
        st.error(f"Failed to load weights. Ensure 'best_model.pth' is in the src folder. Error: {e}")
        return None, None
        
    model.eval()
    
    # Initialize Grad-CAM targeting the last conv layer of EfficientNet
    target_layer = model.backbone.features[-1]
    cam = GradCAM(model, target_layer)
    
    return model, cam

model, cam_extractor = load_model_and_cam()

# ==========================================
# IMAGE PROCESSING PIPELINE
# ==========================================
def process_image(image):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    return transform(image).unsqueeze(0)

# ==========================================
# SIDEBAR & UPLOADER
# ==========================================
with st.sidebar:
    st.header("Upload X-RAY Image")
    uploaded_file = st.file_uploader("Select a Chest X-Ray (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    st.markdown("---")
    st.markdown("**Model Specs:**")
    st.markdown("- **Arch:** EfficientNet-B0")
    st.markdown("- **Explainability:** Grad-CAM")

# ==========================================
# MAIN EXECUTION
# ==========================================
if uploaded_file is not None and model is not None:
    # Read the image in memory
    image = Image.open(uploaded_file).convert('RGB')
    
    # Layout columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original X-Ray")
        st.image(image, use_container_width=True)

    with st.spinner("Analyzing scan..."):
        # Format the tensor
        input_tensor = process_image(image)
        
        # 1. Get Prediction
        prob, binary_label = cam_extractor.get_prediction(input_tensor)
        
        # 2. Generate Heatmap
        heatmap = cam_extractor.generate_heatmap(input_tensor)
        
        # 3. Create the Overlay purely in memory
        overlay_img = overlay_heatmap_from_array(image, heatmap, alpha=0.4)
        
    with col2:
        st.subheader("Grad-CAM Attention Map")
        st.image(overlay_img, use_container_width=True)
        
    # ==========================================
    # DIAGNOSTIC REPORT
    # ==========================================
    st.markdown("---")
    st.subheader("Diagnostic Report")
    
    # Format the metrics
    confidence_percent = prob * 100 if binary_label == 1 else (1 - prob) * 100
    diagnosis_text = "Positive (Pleural Effusion Detected)" if binary_label == 1 else "Negative (Normal)"
    
    # Metric UI
    m1, m2 = st.columns(2)
    m1.metric("Model Prediction", diagnosis_text)
    m2.metric("Confidence", f"{confidence_percent:.1f}%")
    
    # Alert banners
    if binary_label == 1:
        st.error("⚠️ The model detected features consistent with Pleural Effusion. Please review the highlighted regions in the Grad-CAM map.")
    else:
        st.success("✅ The model indicates a low probability of Pleural Effusion.")