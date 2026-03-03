import streamlit as st
import numpy as np
from PIL import Image
import base64
from Models import detection


color_samples = [
    (255, 255, 0),   # Yellow – bright and contrasts well with black
    (0, 255, 255),   # Cyan – highly visible, clean contrast
    (255, 200, 100), # Light orange – warm and readable
    (180, 255, 180), # Pale green – soft and effective
    (255, 128, 255), # Light magenta – vibrant but not dark
]



# Predefined Data Dictionary    
data = {
    'Apple Banana Orange': {
        "class_names": ['Apple', 'Banana', 'Orange'],
        "weights_name": "apple_banana_orange.pt"
    },
    'Bear': {
        "class_names": ['Bear'],
        "weights_name": "bear.pt"
    },
    'Brain Tumor': {
        "class_names": ['Brain Tumor'],
        "weights_name": "brain_tumor.pt"
    },
    'Number Plate': {
        "class_names": ['Number Plate'],
        "weights_name": "number_plate.pt"
    },
    'Pothole': {
        "class_names": ['Pothole'],
        "weights_name": "pothole.pt"
    },
    'Paper Rock Scissors': {
        "class_names": ['Paper', 'Rock', 'Scissors'],
        "weights_name": "rock_paper_scissor.pt"
    }
}


# Page config
st.set_page_config(page_title="Object Detection App", layout="wide")

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return encoded

bg_image_path = "img.jpg"  # Adjust path as needed
bg_image_encoded = get_base64_image(bg_image_path)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bg_image_encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# Custom CSS
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 50px;
        color: #f0f0f0;
        font-weight: bold;        
        margin-top: -65px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown('<div class="title">Object Detection using Pretrained Models</div>', unsafe_allow_html=True)


# Model selection
selected_model_key = st.selectbox("🔍 Select a Model", list(data.keys()))
model_data = data[selected_model_key]

# Upload image
uploaded_file = st.file_uploader("📸 Upload an Image", type=["jpg", "jpeg", "png"])

# Display layout
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    result_img = None
    counter = None

    col1, col2, col3 = st.columns([1.2, 1, 1.2], gap="large")

    with col1:
        st.image(img, use_container_width=True, caption="📥 Uploaded Image")

    with col2:
        
        # Apply custom CSS for styling the button
        st.markdown("""
            <style>
            div.stButton > button {
                margin-left: 30%;
            }       
            div[data-testid="stSpinner"] {
                margin-left: 30%;
            }
            </style>
        """, unsafe_allow_html=True)

        # Simple button 
        button_clicked = st.button("🚀 Run Detection")

        # Run detection if clicked
        if button_clicked:
            with st.spinner("Detecting objects..."):
                result_img, counter = detection(img, model_data["class_names"], color_samples, model_data["weights_name"])

        if counter is not None:
            st.markdown(
                """
                <div style='margin-left: 30px; font-size: 30px; font-weight: 600;'>
                    📊 Detected Counts:
                </div>
                <br>
                """,
                unsafe_allow_html=True
            )
            for i in range(0, len(model_data["class_names"])):
                st.markdown(
                    f"""
                    <div style='margin-left: 30px; font-size: 28px;'>
                        {model_data['class_names'][i]}: {counter[i]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    with col3:
        if result_img is not None:
            st.image(np.array(result_img), use_container_width=True, caption="📤 Detected Output")
