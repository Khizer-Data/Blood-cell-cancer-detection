<h1 align="center">🧬 Blood Cancer Cell Detection</h1>

<p align="center">
  <img src="https://img.shields.io/badge/AI-Powered-blueviolet?style=flat-square&logo=OpenAI" alt="AI-Powered">
  <img src="https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange?style=flat-square&logo=tensorflow">
  <img src="https://img.shields.io/badge/Data-Microscopic%20Images-9cf?style=flat-square&logo=data">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square">
</p>

<p align="center">
  🔬 A deep learning project for automated detection of blood cancer cells using microscopic image data.<br>
  Combines image processing, neural networks, and medical insight to aid early diagnosis. 📊🧫
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#demo">Demo</a> •
  <a href="#model-architecture">Model</a> •
  <a href="#dataset">Dataset</a> •
  <a href="#usage">Usage</a> •
  <a href="#license">License</a>
</p>

# Blood Cancer Detection: A Deep Learning Approach to Image Classification

![Project Banner](https://github.com/Khizer-Data/Blood-cell-cancer-detection/blob/main/images/banner.png)

## 🚀 Project Vision: Empowering Diagnosis through AI

This project leverages **Convolutional Neural Networks (CNNs)** to detect blood cancer by classifying blood cell images. The goal is to provide an assistive diagnostic tool for early screening, not a replacement for professional medical diagnosis.

> **Note:** This is a **proof-of-concept** project and **not for clinical use**.

---

## 🧠 How It Works: Step-by-Step Overview

### 1. **Data Collection**

Two datasets from Kaggle were used:

* `Blood Cancer Image Dataset`: Cancerous blood cell images
* `Blood Cells Image Dataset`: Healthy blood cell images

We use the Kaggle API to fetch and unzip them in Colab:

```python
!kaggle datasets download -d akhiljethwa/blood-cancer-image-dataset
!kaggle datasets download -d unclesamulus/blood-cells-image-dataset
```

---
![Cancer](https://github.com/Khizer-Data/Blood-cell-cancer-detection/blob/main/images/image1.png)
![Non-cancer](https://github.com/Khizer-Data/Blood-cell-cancer-detection/blob/main/images/image1.1.png)
### 2. **Data Preprocessing**

* **File Unification & Labeling**: Images were renamed and merged.
* **Image Cleaning**: Converted to grayscale, resized to 64x64, and denoised using median blur.

```python
image = cv2.resize(image, (64, 64))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.medianBlur(image, 5)
```

📷 **Before & After Preprocessing**
![Preprocessing Comparison](https://github.com/Khizer-Data/Blood-cell-cancer-detection/blob/main/images/image3.png)
![](https://github.com/Khizer-Data/Blood-cell-cancer-detection/blob/main/images/image3.1.png)


---

**Close -up of a Blood Cell**

![Close-up of a Blood Cell](https://github.com/Khizer-Data/Blood-cell-cancer-detection/blob/main/images/image4.png)

### 3. **Class Imbalance Handling**

The dataset had significantly more healthy cell images. To balance this, we **undersampled** the majority class.

```python
df_cancer = df[df.label == "cancer"]
df_non_cancer = df[df.label == "non-cancer"].sample(n=len(df_cancer))
df_balanced = pd.concat([df_cancer, df_non_cancer])
```

📊 **Class Distribution**


**Before:**

![Before](https://raw.githubusercontent.com/Khizer-Data/Blood-cell-cancer-detection/main/images/image5.png)

**After:**

![After](https://raw.githubusercontent.com/Khizer-Data/Blood-cell-cancer-detection/main/images/image6.png)

---

---

### 4. **Exploratory Data Analysis (EDA)**

* Class distribution plots
* Image samples displayed by class

🧪 **Insight**: Balancing data helps the model treat both classes equally, crucial in medical diagnosis.

---

### 5. **Model Architecture**

We designed a simple but powerful CNN:

```python
model = Sequential([
  Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
  MaxPooling2D(2, 2),
  Conv2D(64, (3, 3), activation='relu'),
  MaxPooling2D(2, 2),
  Flatten(),
  Dense(128, activation='relu'),
  Dropout(0.5),
  Dense(1, activation='sigmoid')
])
```

> 🔍 **Why this matters**: Convolution layers extract features, while max pooling reduces dimensionality and computation.

---

### 6. **Training Strategy: Stratified K-Fold Cross-Validation**

To ensure robustness, we used 5-fold stratified cross-validation. Each fold used a different validation set to reduce overfitting.

```python
skf = StratifiedKFold(n_splits=5)
for train_idx, val_idx in skf.split(image_paths, labels):
    # training and validation logic here
```

📈 **Validation Accuracies**

* Fold 1: 99.75%
* Fold 2: 99.98%
* Fold 3: 100.00%
* Fold 4: 99.87%
* Fold 5: 99.87%

🧮 **Mean Accuracy**: **99.90% ± 0.07**

---

### 7. **Evaluation & Visual Proof**

#### ✅ Confusion Matrix

![Confusion Matrix](https://github.com/Khizer-Data/Blood-cell-cancer-detection/blob/main/images/image7.png)

#### 📉 ROC Curve

![ROC Curve](https://github.com/Khizer-Data/Blood-cell-cancer-detection/blob/main/images/image8.png)

#### 📊 Reliability Diagram

![Reliability Diagram](https://github.com/Khizer-Data/Blood-cell-cancer-detection/blob/main/images/image12.png)

---

## ⚠️ Medical Disclaimer

> This project is for educational purposes only and is not approved for clinical use.
>
> * The model is trained on a specific dataset and not generalizable.
> * Consult a medical professional for diagnosis and treatment.

---

## 🔍 Future Directions

* Test on real-world, diverse medical datasets
* Improve explainability using Grad-CAM
* Experiment with advanced architectures (ResNet, EfficientNet)
* Implement real-time prediction with camera feed

---

## 🤝 Collaborate With Me

I'm open to feedback and collaboration!

📧 Email: **[muhammadkhizerzakir@gmail.com](mailto:muhammadkhizerzakir@gmail.com)**

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
