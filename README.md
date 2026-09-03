# plant-disease-transfer-learning
Plant disease classification using transfer learning with a pretrained ResNet18 model in PyTorch.
# 🌱 Plant Disease Detection Using Transfer Learning

A deep learning project for classifying plant leaf diseases using **transfer learning with a pretrained ResNet18 model** in PyTorch.

The model classifies leaf images into five categories:

* 🥔 Potato — Early Blight
* 🥔 Potato — Healthy
* 🍅 Tomato — Early Blight
* 🍅 Tomato — Late Blight
* 🍅 Tomato — Healthy

---

## 📌 Project Overview

Plant diseases can significantly affect agricultural productivity. Early identification of diseases can help farmers take appropriate action before infections spread.

This project explores how **transfer learning** can be used to build an image classification model efficiently by leveraging a pretrained **ResNet18** network instead of training a convolutional neural network entirely from scratch.

The trained model is also integrated into a **Streamlit web application**, allowing users to upload a leaf image and receive a predicted disease class.

---

## 🎯 Problem Statement

Given an image of a plant leaf, the goal is to classify it into one of five predefined disease/health categories.

This is a **multi-class image classification problem**.

---

## 🧠 Approach

The project follows these steps:

1. Load and preprocess the plant leaf dataset.
2. Split the dataset into training, validation, and testing sets.
3. Load a pretrained ResNet18 model.
4. Freeze the pretrained feature extraction layers.
5. Replace the original classification head with a new classifier containing **5 output classes**.
6. Train the new classifier on the plant disease dataset.
7. Evaluate the model on validation and test data.
8. Save the trained model weights.
9. Deploy the model using Streamlit.

---

## 🏗️ Model Architecture

The project uses **ResNet18**, a convolutional neural network pretrained on ImageNet.

Instead of training the entire network from scratch, the pretrained layers are used as feature extractors.

The original ResNet18 classification layer was replaced with a custom fully connected layer:

```text
ResNet18
   │
   ├── Pretrained convolutional layers
   │
   ├── Feature extraction
   │
   └── Fully Connected Layer
           │
           └── 5 output classes
```

This approach allows the model to take advantage of features learned from a large image dataset.

---

## 📊 Dataset

The project uses the **PlantVillage dataset** containing plant leaf images.

A subset containing five classes was used:

| Class                 | Category |
| --------------------- | -------- |
| Potato___Early_blight | Disease  |
| Potato___healthy      | Healthy  |
| Tomato___Early_blight | Disease  |
| Tomato___Late_blight  | Disease  |
| Tomato___healthy      | Healthy  |

The dataset was divided approximately as follows:

* **70%** — Training
* **15%** — Validation
* **15%** — Testing

The dataset itself is **not included in this repository** because of its size.

---

## ⚙️ Technologies Used

* Python
* PyTorch
* Torchvision
* ResNet18
* Transfer Learning
* Streamlit
* NumPy
* Pillow

---

## 🔄 Image Preprocessing

Input images are resized to:

```text
224 × 224
```

They are then converted to tensors and normalized using the ImageNet normalization values:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

These preprocessing values are used because the ResNet18 model was pretrained using ImageNet.

---

## 📈 Results

The transfer learning model achieved approximately:

| Metric              | Performance |
| ------------------- | ----------: |
| Validation Accuracy |  **96.34%** |
| Test Accuracy       |  **97.53%** |

These results show that a pretrained CNN can perform very well on this classification task with relatively little task-specific training.

> **Note:** The current experiment uses a randomly generated train/validation/test split. For fully reproducible benchmarking, the dataset split should be controlled with a fixed random seed.

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Nerd-master-Akib/plant-disease-transfer-learning.git
```

### 2. Navigate to the project

```bash
cd plant-disease-transfer-learning
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Download the PlantVillage dataset and place the required images in the appropriate dataset directory.

The dataset is intentionally excluded from this repository.

### 5. Train the model

```bash
python train.py
```

### 6. Run prediction

```bash
python predict.py
```

### 7. Launch the Streamlit application

```bash
streamlit run app.py
```

---

## 🌐 Streamlit Application

The project includes a Streamlit interface where users can upload a plant leaf image and receive a predicted class.

### Example workflow

```text
Upload Leaf Image
        ↓
Image Preprocessing
        ↓
ResNet18 Model
        ↓
Prediction
        ↓
Predicted Disease / Healthy Class
```

---

## ⚠️ Limitations

The model was trained on PlantVillage-style images, which are generally cleaner and more controlled than images captured in real agricultural environments.

Therefore, performance may decrease when the model receives:

* Images with complex backgrounds
* Poor-quality photographs
* Different lighting conditions
* Unseen plant species
* Diseases outside the five trained classes

The model will also always choose one of the five available classes, even when an uploaded image does not actually belong to any of them.

---

## 🔮 Future Improvements

Possible improvements include:

* Add more plant species and disease classes
* Use data augmentation
* Fine-tune deeper ResNet layers
* Add confidence/probability visualization
* Implement out-of-distribution detection
* Improve real-world image robustness
* Compare ResNet18 with other pretrained architectures
* Deploy the application online

---

## 👨‍💻 Author

**Akib**

Statistics Student | Machine Learning & AI Enthusiast

This project was developed as part of my progression from traditional machine learning to deep learning and transfer learning.
