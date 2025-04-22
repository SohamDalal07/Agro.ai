import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import gdown

# Load model
model = load_model("shuffuled_model.h5", compile=False, custom_objects={'InputLayer': tf.keras.layers.InputLayer})

# Download large CSV if not exists
if not os.path.exists("shuffled_file.csv"):
    csv_url = "https://drive.google.com/uc?id=1_SMwMKvBZwqk_d_tAnRYU-k8vhaJeUJb"
    gdown.download(csv_url, "shuffled_file.csv", quiet=False)

# Load data
df = pd.read_csv("shuffled_file.csv")
supplement_df = pd.read_csv("supplement_info.csv")
info_df = pd.read_csv("disease_info.csv", encoding='latin-1')

# Preprocessing
le = LabelEncoder()
le.fit(df['label'].values)
feature_extractor = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

scaler = StandardScaler()
X_train_features = df.drop(columns=['image_name', 'label']) if 'image_name' in df.columns else df.drop(columns=['label'])
scaler.fit(X_train_features)

def normalize_name(name):
    return str(name).strip().lower().replace(" ", "").replace(":", "").replace("|", "").replace("_", "")

# Streamlit UI
st.set_page_config(page_title="Agro.ai - Crop Disease Detector")
st.title("🌿 Agro.ai - Crop Disease Detection")
st.markdown("Upload a crop image to identify the disease and get remedies.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).resize((224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = feature_extractor.predict(img_array)[0].reshape(1, -1)
    features_scaled = scaler.transform(features)
    features_scaled = np.expand_dims(features_scaled, axis=2)

    pred = model.predict(features_scaled)
    predicted_index = np.argmax(pred)
    predicted_label = le.inverse_transform([predicted_index])[0]

    st.subheader(f"🧪 Prediction: **{predicted_label}**")

    predicted_clean = normalize_name(predicted_label)
    supplement_df['normalized'] = supplement_df['disease_name'].apply(normalize_name)
    info_df['normalized'] = info_df['disease_name'].apply(normalize_name)

    matched_supp = supplement_df[supplement_df['normalized'] == predicted_clean]
    matched_info = info_df[info_df['normalized'] == predicted_clean]

    if not matched_supp.empty:
        supp = matched_supp.iloc[0]
        st.markdown(f"**💊 Supplement:** [{supp['supplement name']}]({supp['buy link']})")

    if not matched_info.empty:
        inf = matched_info.iloc[0]
        st.markdown(f"**📖 Description:** {inf['description']}")
        st.markdown(f"**🛠 Possible Steps:** {inf['Possible Steps']}")
        if pd.notnull(inf['image_url']):
            st.image(inf['image_url'], caption="Related Disease Image", use_column_width=True)
