import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Skin Cancer Detection System",
    page_icon="🩺",
    layout="wide"
)

# ==========================
# CUSTOM CSS
# ==========================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}

.main-title {
    text-align:center;
    color:white;
    font-size:40px;
    font-weight:bold;
}

.sub-title {
    text-align:center;
    color:#cbd5e1;
    font-size:18px;
}

.pred-box {
    background-color:#14532d;
    padding:15px;
    border-radius:10px;
    color:white;
    font-size:22px;
    font-weight:bold;
}

.conf-box {
    background-color:#1e3a8a;
    padding:15px;
    border-radius:10px;
    color:white;
    font-size:20px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div class='main-title'>🩺 Skin Cancer Detection System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Upload a skin lesion image to predict the disease class</div>",
    unsafe_allow_html=True
)

st.write("")

# ==========================
# MODEL LOAD
# ==========================
MODEL_PATH = "best_cnn_model_image.keras"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("✅ Model Loaded Successfully")
except Exception as e:
    st.error(f"❌ Model Loading Error: {str(e)}")
    st.stop()

# ==========================
# CLASS NAMES
# ==========================
class_names = [
    "Acne",
    "Actinic Keratosis",
    "Atopic",
    "Basal Cell Carcinoma",
    "BCC",
    "Benign",
    "Dermatofibroma",
    "Malignant",
    "Melanoma",
    "Nevus",
    "Non-Neoplastic",
    "Pigmented Benign Keratosis",
    "Seborrheic Keratosis",
    "Squamous Cell Carcinoma",
    "Vascular Lesion"
]

# ==========================
# FILE UPLOAD
# ==========================
uploaded_file = st.file_uploader(
    "Upload Skin Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    # ==========================
    # PREPROCESS IMAGE
    # ==========================
    img = image.resize((128, 128))

    img_array = np.array(
        img,
        dtype=np.float32
    ) / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # ==========================
    # PREDICTION
    # ==========================
    prediction = model.predict(
        img_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = prediction[predicted_index] * 100

    with col2:

        st.markdown(
            f"""
            <div class='pred-box'>
            Prediction: {predicted_class}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        st.markdown(
            f"""
            <div class='conf-box'>
            Confidence: {confidence:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

    # ==========================
    # TABLE
    # ==========================
    result_df = pd.DataFrame({
        "Disease": class_names,
        "Match %": prediction * 100
    })

    result_df = result_df.sort_values(
        by="Match %",
        ascending=False
    )

    st.write("")
    st.subheader("📊 Disease Match Percentage")

    st.dataframe(
        result_df.style.format({
            "Match %": "{:.2f}"
        }),
        use_container_width=True
    )

    # ==========================
    # CHART
    # ==========================
    st.subheader("📈 Probability Distribution")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        result_df["Disease"],
        result_df["Match %"]
    )

    ax.set_xlabel(
        "Match Percentage (%)"
    )

    ax.set_ylabel(
        "Disease"
    )

    plt.tight_layout()

    st.pyplot(fig)
