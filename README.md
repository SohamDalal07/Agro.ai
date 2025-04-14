# Agro.ai
# 🌿 Crop Disease Detection using AI

This project is an AI-powered solution designed to identify crop diseases from leaf images, leveraging deep learning and machine learning models. 
It also provides disease-specific supplement recommendations and treatment information, making it an end-to-end intelligent tool for farmers and agricultural experts.


## 🧠 Project Objective

- Detect plant diseases accurately from leaf images using **Convolutional Neural Networks (CNN)**
- Extract image features using **MobileNetV2**
- Compare performance between **Random Forest** and **CNN**
- Recommend appropriate **supplements** based on disease detection
- Provide an **automated interface** that bridges detection with actionable steps

---

## 📂 Dataset & Preprocessing

- **Source**: [Kaggle Plant Disease Dataset](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset)
- **Total Images**: ~87,000 RGB images  
- **Categories**: 38 plant disease classes (e.g., `Tomato_Leaf_Mold`, `Apple_scab`, etc.)  
- **Preprocessing**:
  - Used **MobileNetV2** (pre-trained CNN) to extract 1280-dimensional feature vectors
  - Created a **CSV file**, where each row corresponds to a plant image with its extracted features

---

## 🌳 Machine Learning Models

### 🔹 Random Forest
- Ensemble method that builds multiple decision trees
- **Advantages**:
  - Handles high-dimensional data
  - Resistant to overfitting
- **Usage**: Served as a baseline model before CNN
- **Result**: Gave moderate accuracy and highlighted the strength of deep learning models

### 🔹 CNN Model Architecture

```python
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y.shape[1], activation='softmax')  # Number of classes
])
```

---

## 🔍 Disease Detection Workflow

### 🧪 Feature Extraction

```python
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)
features = feature_extractor.predict(img_array)[0].reshape(1, -1)
```

---

## 💊 Supplement & Treatment Recommendation

- **Supplement Recommendation**:
  - CSV file maps diseases to supplements and purchase links
  - Flexible string matching retrieves relevant results

- **Disease Information Retrieval**:
  - CSV file includes:
    - Disease description
    - Treatment steps
    - Illustrative image links

---

## 💻 Interface

The interface displays:
- Detected disease name
- Description and treatment info
- Recommended supplements and links

---

## 📈 Model Evaluation

- Confusion matrix
- Accuracy metrics to evaluate model performance
- Improvement visualized from Random Forest to CNN

---

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- MobileNetV2
- Pandas, NumPy
- Jupyter Notebooks

---


---

Let me know if you want this exported as a `.md` file or want a badge section or deployment section added!


