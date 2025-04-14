from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import gdown  # Only needed if using Google Drive for large CSVs

app = Flask(__name__)

# Model is directly in project folder
model = load_model("shuffuled_model.h5", compile=False, custom_objects={'InputLayer': InputLayer})

# Check & download large CSV from Google Drive if not exists
if not os.path.exists("shuffled_file.csv"):
    csv_url = "https://drive.google.com/uc?id=1_SMwMKvBZwqk_d_tAnRYU-k8vhaJeUJb"
    gdown.download(csv_url, "shuffled_file.csv", quiet=False)

# Load CSVs
df = pd.read_csv("shuffled_file.csv")
supplement_df = pd.read_csv("supplement_info.csv")
info_df = pd.read_csv("disease_info.csv", encoding='latin-1')

# Preprocessing setup
le = LabelEncoder()
le.fit(df['label'].values)
feature_extractor = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def normalize_name(name):
    return str(name).strip().lower().replace(" ", "").replace(":", "").replace("|", "").replace("_", "")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    supplement = None
    info = None
    img_url = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            img = Image.open(file).resize((224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            features = feature_extractor.predict(img_array)[0].reshape(1, -1)
            X_train_features = df.drop(columns=['image_name', 'label']) if 'image_name' in df.columns else df.drop(columns=['label'])

            scaler = StandardScaler()
            scaler.fit(X_train_features)
            features_scaled = scaler.transform(features)
            features_scaled = np.expand_dims(features_scaled, axis=2)

            pred = model.predict(features_scaled)
            predicted_index = np.argmax(pred)
            predicted_label = le.inverse_transform([predicted_index])[0]

            prediction = predicted_label
            predicted_clean = normalize_name(predicted_label)

            supplement_df['normalized'] = supplement_df['disease_name'].apply(normalize_name)
            info_df['normalized'] = info_df['disease_name'].apply(normalize_name)

            matched_supp = supplement_df[supplement_df['normalized'] == predicted_clean]
            matched_info = info_df[info_df['normalized'] == predicted_clean]

            if not matched_supp.empty:
                supp = matched_supp.iloc[0]
                supplement = {
                    "name": supp['supplement name'],
                    "link": supp['buy link']
                }

            if not matched_info.empty:
                inf = matched_info.iloc[0]
                info = {
                    "description": inf['description'],
                    "steps": inf['Possible Steps']
                }
                if pd.notnull(inf['image_url']):
                    img_url = inf['image_url']

    return render_template("index.html", prediction=prediction, supplement=supplement, info=info, img_url=img_url)

if __name__ == "__main__":
    app.run(debug=True)
