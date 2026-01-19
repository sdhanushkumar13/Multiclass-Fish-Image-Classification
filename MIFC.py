import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Fish Classification",
    layout="wide"
)

model = tf.keras.models.load_model("models/InceptionV3_best.h5")

class_names = ['animal fish', 'animal fish bass', 'fish sea_food black_sea_sprat', 'fish sea_food gilt_head_bream',
               'fish sea_food hourse_mackerel', 'fish sea_food red_mullet', 'fish sea_food red_sea_bream',
               'fish sea_food sea_bass', 'fish sea_food shrimp', 'fish sea_food striped_red_mullet', 'fish sea_food trout']


st.markdown(
    "<h1 style='text-align: center;'>🦈 Multiclass Fish Image Classification</h1>",
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Fish Image")
    st.write("Upload a fish image to classify its species.")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=300)

with col2:
    st.subheader("📊 Prediction Results")

    if uploaded_file is not None:
        resized_image = image.resize((224, 224))
        img_array = np.array(resized_image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        class_index = np.argmax(predictions)
        confidence = float(np.max(predictions))

        st.success(f"Predicted Class: {class_names[class_index]}")
        st.info(f"Confidence Score: {confidence:.2f}")
    else:
        st.write("⬅️ Upload an image to see prediction results.")
