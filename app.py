import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd

st.set_page_config(page_title="Kissan+ | Plant Disease Detection", page_icon="🌿")
st.title("🌿 Kissan+ | Multi-Disease Detection (10 Classes)")
st.write("Upload a leaf image to detect its disease and view detailed probability analysis.")

# Load trained 10-class model
model = load_model("mini_plant_model.keras")

# Define class names (must match training order)
classes = [
    "Tomato_Late_Blight",
    "Potato_Early_Blight",
    "Tomato_Healthy",
    "Corn_Gray_Leaf_Spot",
    "Apple_Black_Rot",
    "Grape_Black_Rot",
    "Pepper_Bacterial_Spot",
    "Peach_Bacterial_Spot",
    "Strawberry_Leaf_Scorch",
    "Tomato_Target_Spot"
]

uploaded_file = st.file_uploader("📸 Upload a leaf image...", type=["jpg","jpeg","png"])

if uploaded_file:
    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(128,128))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)[0]
    predicted_index = np.argmax(predictions)
    result = classes[predicted_index]
    confidence = predictions[predicted_index] * 100

    # Show prediction
    st.image(img, caption=f"Predicted: {result}", use_column_width=True)
    st.success(f"✅ Predicted Disease: **{result}**")
    st.info(f"📊 Confidence: **{confidence:.2f}%**")

    # Show probability bar chart
    st.write("### 🔍 Probability Distribution")
    df = pd.DataFrame({
        "Disease": classes,
        "Probability (%)": [p*100 for p in predictions]
    }).sort_values("Probability (%)", ascending=True)

    st.bar_chart(df.set_index("Disease"))
