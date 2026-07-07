import streamlit as st
import os
import sys
import json
from PIL import Image
from dotenv import load_dotenv

# Ensure local source directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.classifier import predict_defect
from src.agent import generate_maintenance_plan

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="AI Defect Inspector & Maintenance Planner",
    page_icon="🔧",
    layout="wide"
)

# Custom CSS for premium dark-mode industrial theme
st.markdown("""
<style>
    .reportview-container {
        background: #0F172A;
    }
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #F59E0B, #EF4444);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #94A3B8;
        margin-bottom: 1.8rem;
    }
    .metric-card {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #334155;
        margin-bottom: 12px;
    }
    .severity-high {
        color: #EF4444;
        font-weight: bold;
        background-color: #450A0A;
        padding: 4px 8px;
        border-radius: 4px;
        border: 1px solid #991B1B;
    }
    .severity-medium {
        color: #F59E0B;
        font-weight: bold;
        background-color: #451A03;
        padding: 4px 8px;
        border-radius: 4px;
        border: 1px solid #92400E;
    }
    .severity-low {
        color: #10B981;
        font-weight: bold;
        background-color: #064E3B;
        padding: 4px 8px;
        border-radius: 4px;
        border: 1px solid #065F46;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🔧 Autonomous Defect Inspector</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Multi-Model Deep Learning Image Classifier (ResNet & EfficientNet) & Generative AI Maintenance Planner (LangChain RAG)</p>', unsafe_allow_html=True)

# Sidebar Info & Dynamic API Key input
api_key_from_env = os.getenv("GROQ_API_KEY")

with st.sidebar:
    st.header("🔧 Vision \& RAG Auditor")
    
    if not api_key_from_env:
        st.subheader("🔑 API Authentication")
        user_api_key = st.text_input("Enter Groq API Key:", type="password", help="Get a free key from console.groq.com")
        if user_api_key:
            os.environ["GROQ_API_KEY"] = user_api_key
            st.success("API Key applied locally!")
            st.rerun()
        else:
            st.warning("Please enter your Groq API Key to proceed.")
            
    st.write("---")
    st.write("AegisVision coordinates process safety audits:")
    st.markdown("- **📷 Visual Inspection**: Multi-model deep learning (ResNet / EfficientNet) classifying surface defects.")
    st.markdown("- **📄 Maintenance Planner**: RAG pipeline retrieving safety guidelines and compiling Markdown reports.")
    st.write("---")
    st.caption("Powered by PyTorch \& LangChain.")

# Check if API key was loaded
if not os.getenv("GROQ_API_KEY"):
    st.error("⚠️ Groq API Key is missing. Please enter your `GROQ_API_KEY` in the sidebar to start the app.")
    st.stop()

# Layout: Split screen
col_left, col_right = st.columns([1, 2])

with col_left:
    st.header("📷 Visual Inspection")
    
    # Image upload
    uploaded_file = st.file_uploader("Upload an image of the equipment or surface:", type=["jpg", "jpeg", "png"])
    
    # Sample images selector
    st.write("Or try a generated sample image from the validation set:")
    samples_dir = "dataset/val"
    sample_choice = None
    
    if os.path.exists(samples_dir):
        # We find one image path for each folder in val
        sample_paths = {}
        for folder in ['crack', 'rust', 'wear', 'normal']:
            folder_path = os.path.join(samples_dir, folder)
            if os.path.exists(folder_path):
                files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
                if files:
                    sample_paths[folder.capitalize()] = os.path.join(folder_path, files[0])
                    
        # Multi-button or selectbox selection
        options = ["-- None --"] + list(sample_paths.keys())
        choice = st.selectbox("Select a sample defect image:", options)
        if choice != "-- None --":
            sample_choice = sample_paths[choice]
            
    # Process Image Choice
    image_to_process = None
    if uploaded_file is not None:
        image_to_process = uploaded_file
    elif sample_choice is not None:
        image_to_process = sample_choice
        
    if image_to_process is not None:
        # Display image
        img = Image.open(image_to_process)
        st.image(img, caption="Inspected Component", use_column_width=True)
        
        # Run classification
        with st.spinner("🧠 PyTorch Multi-Model CNN analyzing visual features..."):
            pred_result = predict_defect(image_to_process)
            
        # Display Prediction Cards
        st.subheader("📊 CNN Classification Results")
        
        defect_class = pred_result["class"]
        confidence = pred_result["confidence"]
        severity = pred_result["severity"]
        method = pred_result["method"]
        
        # Format severity badge
        severity_class = f"severity-{severity.lower()}"
        
        st.markdown(f"""
        <div class="metric-card">
            <p>👁️ <strong>Detected Class:</strong> <span style="font-size: 1.15rem; color: #F59E0B; text-transform: uppercase;">{defect_class}</span></p>
            <p>📈 <strong>Confidence Score:</strong> {confidence:.2%}</p>
            <p>🚨 <strong>Severity Classification:</strong> <span class="{severity_class}">{severity}</span></p>
            <p>🔬 <strong>Analysis Method:</strong> {method}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar for confidence
        st.progress(float(confidence))

        # Benchmarking Comparison Expander
        comparison_file = 'models/model_comparison.json'
        if os.path.exists(comparison_file):
            with st.expander("📊 Model Benchmark Study (Parameter & Latency Trade-offs)"):
                try:
                    with open(comparison_file, 'r', encoding='utf-8') as f:
                        metrics_data = json.load(f)
                    
                    table_md = """
| Architecture | Total Params | Val Accuracy | Latency/Img |
| :--- | :---: | :---: | :---: |
"""
                    for m in metrics_data:
                        table_md += f"| **{m['model_name']}** | {m['total_params']:,} | {m['val_acc']:.2f}% | {m['avg_latency_ms']:.2f} ms |\n"
                    
                    st.markdown(table_md)
                    st.caption("Note: Benchmarked on CPU to evaluate edge deployment viability.")
                except Exception as e:
                    st.error(f"Error loading benchmarking report: {e}")

with col_right:
    st.header("📄 Maintenance Ticket & Action Plan")
    
    if image_to_process is not None:
        # Run LangChain RAG pipeline
        with st.spinner("🎙️ LangChain retrieving safety guidelines and compiling plan..."):
            plan = generate_maintenance_plan(defect_class, confidence, severity)
            
        # Display Plan
        st.markdown(plan)
        
        # Download button
        st.download_button(
            label="📥 Download Maintenance Plan (.md)",
            data=plan,
            file_name=f"maintenance_plan_{defect_class}_{severity.lower()}.md",
            mime="text/markdown"
        )
    else:
        st.info("Upload an image on the left or select a sample image to generate the autonomous maintenance plan.")
