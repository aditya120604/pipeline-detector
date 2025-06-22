import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Configure page
st.set_page_config(
    page_title="Sea Pipeline Corrosion Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model function
@st.cache_resource
def load_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load labels
@st.cache_data
def load_labels():
    try:
        with open("labels.txt", "r") as f:
            labels = [line.strip().split(" ", 1)[1] for line in f.readlines()]
        return labels
    except Exception as e:
        st.error(f"Error loading labels: {str(e)}")
        return ["Normal", "Corroded"]

# Prediction function
def predict_image(interpreter, image, labels):
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preprocess image
    input_shape = input_details[0]['shape']
    image = image.resize((input_shape[1], input_shape[2]))
    image_array = np.array(image, dtype=np.float32)
    
    # Normalize image
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image_array)
    
    # Run inference
    interpreter.invoke()
    
    # Get prediction
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data[0])
    confidence = float(output_data[0][prediction])
    
    return labels[prediction], confidence

# Sidebar navigation
st.sidebar.title("🚢 Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["🏠 Home", "🔍 Pipeline Inspector", "📊 About Model", "📖 Instructions", "ℹ️ Info"]
)

# Main content based on selected page
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🔍 Sea Pipeline Corrosion Detector</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://via.placeholder.com/400x200/1f77b4/white?text=Pipeline+Monitoring", 
                caption="Advanced AI-Powered Pipeline Inspection")
    
    st.markdown("""
    <div class="info-box">
        <h3>🌊 Protecting Our Underwater Infrastructure</h3>
        <p>This application uses advanced machine learning to detect corrosion in underwater pipelines, 
        helping prevent environmental disasters and maintain critical infrastructure.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>🤖 AI-Powered</h4>
            <p>Trained using Google's Teachable Machine with specialized pipeline imagery</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>⚡ Fast Detection</h4>
            <p>Get instant results with high accuracy corrosion detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-box">
            <h4>🛡️ Prevention Focus</h4>
            <p>Early detection helps prevent costly repairs and environmental damage</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "🔍 Pipeline Inspector":
    st.markdown('<h1 class="main-header">🔍 Pipeline Inspector</h1>', unsafe_allow_html=True)
    
    # Load model and labels
    interpreter = load_model()
    labels = load_labels()
    
    if interpreter is None:
        st.error("❌ Model could not be loaded. Please ensure 'model.tflite' is in the correct directory.")
        st.stop()
    
    st.markdown("""
    <div class="info-box">
        <h3>📸 Upload Pipeline Image</h3>
        <p>Upload an image of a pipeline section to check for corrosion. 
        Supported formats: JPG, JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of the pipeline section you want to inspect"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Pipeline Image", use_column_width=True)
        
        with col2:
            st.markdown("### 🔄 Analysis in Progress...")
            
            with st.spinner("Analyzing image for corrosion..."):
                try:
                    # Make prediction
                    prediction, confidence = predict_image(interpreter, image, labels)
                    
                    # Display results
                    st.markdown("### 📊 Analysis Results")
                    
                    if prediction == "Normal":
                        st.markdown(f"""
                        <div class="success-box">
                            <h4>✅ {prediction}</h4>
                            <p><strong>Confidence:</strong> {confidence:.2%}</p>
                            <p>The pipeline section appears to be in good condition with no visible corrosion.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="danger-box">
                            <h4>⚠️ {prediction} Detected</h4>
                            <p><strong>Confidence:</strong> {confidence:.2%}</p>
                            <p>Corrosion has been detected in this pipeline section. Immediate inspection and maintenance may be required.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Confidence meter
                    st.markdown("### 📈 Confidence Level")
                    st.progress(confidence)
                    st.write(f"Model confidence: {confidence:.2%}")
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")

elif page == "📊 About Model":
    st.markdown('<h1 class="main-header">📊 About the Model</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>🤖 Model Architecture</h3>
        <p>This model was trained using Google's Teachable Machine platform, which provides 
        an accessible way to create machine learning models for image classification.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Model Specifications")
        st.markdown("""
        - **Type**: Image Classification
        - **Framework**: TensorFlow Lite
        - **Classes**: 2 (Normal, Corroded)
        - **Input**: RGB Images
        - **Platform**: Teachable Machine
        """)
        
        st.markdown("### 🏗️ Training Process")
        st.markdown("""
        1. **Data Collection**: Gathered pipeline images
        2. **Labeling**: Classified as Normal or Corroded
        3. **Training**: Used Teachable Machine interface
        4. **Export**: Generated TensorFlow Lite model
        5. **Deployment**: Integrated into Streamlit app
        """)
    
    with col2:
        st.markdown("### 📈 Performance Metrics")
        st.info("Model performance depends on training data quality and variety")
        
        st.markdown("### 🎯 Use Cases")
        st.markdown("""
        - **Preventive Maintenance**: Early corrosion detection
        - **Safety Inspections**: Regular pipeline monitoring
        - **Cost Reduction**: Avoid major repairs
        - **Environmental Protection**: Prevent leaks and spills
        """)

elif page == "📖 Instructions":
    st.markdown('<h1 class="main-header">📖 How to Use</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>🚀 Quick Start Guide</h3>
        <p>Follow these simple steps to inspect your pipeline images for corrosion:</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📝 Basic Usage", "📸 Image Guidelines", "⚡ Tips & Tricks"])
    
    with tab1:
        st.markdown("### Step-by-Step Instructions")
        st.markdown("""
        1. **Navigate** to the "🔍 Pipeline Inspector" page
        2. **Upload** your pipeline image using the file uploader
        3. **Wait** for the analysis to complete (usually takes a few seconds)
        4. **Review** the results and confidence score
        5. **Take action** based on the detection results
        """)
        
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Important Notes</h4>
            <ul>
                <li>Ensure your image is clear and well-lit</li>
                <li>The model works best with close-up pipeline shots</li>
                <li>Results should be verified by qualified personnel</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 📸 Image Quality Guidelines")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ✅ Good Images")
            st.markdown("""
            - Clear, high-resolution photos
            - Good lighting conditions
            - Pipeline fills most of the frame
            - Minimal background distractions
            - Sharp focus on pipeline surface
            """)
        
        with col2:
            st.markdown("#### ❌ Avoid These")
            st.markdown("""
            - Blurry or out-of-focus images
            - Very dark or overexposed photos
            - Pipeline too small in frame
            - Heavy shadows or reflections
            - Multiple pipelines in one image
            """)
    
    with tab3:
        st.markdown("### ⚡ Tips for Best Results")
        st.markdown("""
        - **Lighting**: Use natural daylight or proper underwater lighting
        - **Distance**: Keep camera 1-3 feet from pipeline surface
        - **Angle**: Take photos perpendicular to pipeline surface
        - **Coverage**: Capture different sections for comprehensive analysis
        - **Documentation**: Keep records of inspection dates and locations
        """)

elif page == "ℹ️ Info":
    st.markdown('<h1 class="main-header">ℹ️ Additional Information</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🔒 Disclaimer", "🛠️ Technical Details", "📞 Support"])
    
    with tab1:
        st.markdown("""
        <div class="warning-box">
            <h3>⚠️ Important Disclaimer</h3>
            <p>This AI model is designed to assist in pipeline inspection but should not replace 
            professional assessment. Always consult with qualified marine engineers and pipeline 
            specialists for critical infrastructure decisions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 Model Limitations")
        st.markdown("""
        - Results are based on training data and may not cover all corrosion types
        - Environmental factors can affect detection accuracy
        - False positives and negatives are possible
        - Regular model updates may be needed for optimal performance
        """)
    
    with tab2:
        st.markdown("### 🛠️ Technical Specifications")
        st.markdown("""
        - **Model Format**: TensorFlow Lite (.tflite)
        - **Input Size**: Varies based on training (typically 224x224)
        - **Color Space**: RGB
        - **Inference Time**: < 1 second
        - **Memory Usage**: Optimized for edge deployment
        """)
        
        st.markdown("### 📁 Required Files")
        st.code("""
        project/
        ├── app.py (this file)
        ├── model.tflite
        └── labels.txt
        """)
    
    with tab3:
        st.markdown("### 📞 Need Help?")
        st.markdown("""
        If you encounter any issues or have questions about the pipeline corrosion detector:
        
        - Check that all required files are in the correct location
        - Ensure your images meet the quality guidelines
        - Try different images if results seem inconsistent
        - Contact your system administrator for technical support
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 2rem;'>
    <p>🔍 Sea Pipeline Corrosion Detector | Powered by AI 🤖</p>
    <p><em>Protecting underwater infrastructure through intelligent monitoring</em></p>
</div>
""", unsafe_allow_html=True)
