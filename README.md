# 🛰️ Image-Based Geolocation Prediction on IIIT-H Campus
*IIIT Hyderabad | Spring 2025*

---
## 📌 Project Overview

This project aims to predict info from images taken across the IIIT Hyderabad campus. Using annotated data with spatial and temporal metadata, a CNN-based model was developed to learn visual and spatial patterns for accurate geolocation and orientation estimation on unseen images.
- **Latitude** (scaled)
- **Longitude** (scaled)
- **Camera Angle**
- **Region ID** (categorical: 1–15)

The objective is to develop a robust deep learning model capable of both regression and classification based on visual input alone.

---

## 🧠 Task Description

- **Data Collection**:
  - Captured and annotated over 3,000 images across 15 predefined regions on campus.
  - Each image is tagged with metadata: GPS coordinates, timestamp, orientation angle, and region ID.

- **Dataset Features**:
  - `Image`
  - `Latitude` *(scaled)*
  - `Longitude` *(scaled)*
  - `Angle`
  - `Region ID` (1–15)

- **Modeling Approach**:
  - Designed a **CNN-based architecture** for:
    - **Regression**: Latitude, Longitude, Angle
    - **Classification**: Region ID
  - Compared performance of multi-output joint models vs. task-specific models.

- **Evaluation Metrics**:
  - Mean Squared Error (MSE) for latitude, longitude, and angle.
  - Accuracy (up to 6 decimal places) for region classification.

---

## 📊 Results

| Task                | Metric | Value     |
|---------------------|--------|-----------|
| Latitude Prediction | MSE    | < 0.25    |
| Longitude Prediction| MSE    | < 0.27    |
| Angle Prediction    | MAAE    | < 0.05    |
| Region ID Classification | Accuracy | **91.%** |

---

## 🧰 Tools & Techniques

- Python, NumPy, Pandas
- TensorFlow / PyTorch
- CNNs (Convolutional Neural Networks)
- Transfer Learning (ResNet-50 and ImageNet)
- Multi-output Learning (Regression + Classification)
- Data Augmentation and Preprocessing
- Matplotlib & Seaborn for visualization

---

## 🎯 Key Takeaways

- Applied deep learning to a real-world **geospatial computer vision** task.
- Developed an end-to-end pipeline from **data collection and labeling** to **model training and evaluation**.
- Built a model that can **localize an image** and estimate **camera orientation** using only visual content.
- Gained hands-on experience with **multi-task learning** in CNNs.

---

## 🙋‍♂️ Author

**Jagankrishna Nallasingu**  
B.Tech CSE, IIIT Hyderabad  (2022-2026)
[LinkedIn](https://www.linkedin.com/in/jagankrishna-nallasingu-0725b4268/)
