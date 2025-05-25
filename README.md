# 📦 X-Ray Cargo Anomaly Detection using Siamese Network

This project focuses on detecting anomalies in X-ray cargo images using a Siamese Neural Network built with a ResNet-18 backbone. The model distinguishes between normal (negative) and anomalous (positive) cargo samples by learning image similarity through one-shot learning.

---

## 📁 Dataset

Two sets of images were used for training and evaluation, hosted on Google Drive:

- 🔴 **Positive (Anomalous) Samples**: [Download](https://drive.google.com/file/d/1NK1DWLMztROwRkJIlYAnexWLgyFv_gDF/view)
- 🟢 **Negative (Normal) Samples**: [Download](https://drive.google.com/file/d/18QJyRNVDG6jguNmV04GRuZM98IGdizUb/view)

All images are in `.png` format.

---

## 🧠 Model Overview

- **Architecture**: Siamese Neural Network
- **Backbone**: Pre-trained ResNet-18
- **Loss Function**: Contrastive Loss
- **Similarity Metric**: Euclidean Distance

The network learns embeddings that map similar images closer and dissimilar ones farther apart in feature space.

---

## 🛠️ Technologies Used

- Python
- PyTorch
- Google Colab
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

---

## 🚀 How to Run the Project

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/xray-anomaly-detection.git
   cd xray-anomaly-detection
