import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Pipeline Corrosion Detection",
    page_icon="🛠️",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
    }
    .main {
        background-color: #1e1e1e;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(0,0,0,0.2);
        margin-top: -60px;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    .stButton>button {
        background-color: #004080;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        padding-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🔍 About")
st.sidebar.markdown("""
This tool uses a trained **machine learning model** to detect whether a pipeline image is **Normal** or **Corroded**.

**Instructions:**
- Upload a clear image of a pipeline
- Wait for the model to process it
- Get the prediction and confidence score
""")

# Load TensorFlow Lite model
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

interpreter, input_details, output_details = load_tflite_model()

# Inference
def predict(image: Image.Image):
    image = image.resize((224, 224))  # Resize to TM standard
    image_np = np.asarray(image).astype(np.float32)
    normalized = (image_np / 127.5) - 1
    input_data = np.expand_dims(normalized, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    return output

# Main section
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.title("Pipeline Corrosion Detection")
st.subheader("Upload an image to detect pipeline condition")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    output = predict(image)
    class_names = ["Normal", "Corroded"]
    predicted_index = int(np.argmax(output))
    predicted_label = class_names[predicted_index]
    confidence = float(output[predicted_index]) * 100

    st.success(f"Prediction: **{predicted_label}**")
    st.info(f"Confidence: **{confidence:.2f}%**")

st.markdown("</div>", unsafe_allow_html=True)
