# 📦 X-Ray Cargo Anomaly Detection using Siamese Network

This project focuses on detecting anomalies in X-ray cargo images using a Siamese Neural Network with a ResNet-18 backbone. The model differentiates between normal (negative) and anomalous (positive) cargo images through one-shot learning using contrastive loss and Euclidean distance.

---

## 📁 Dataset

Two datasets were used and are available on Google Drive:

- 🔴 **Positive (Anomalous) Samples**: [Download](https://drive.google.com/file/d/1NK1DWLMztROwRkJIlYAnexWLgyFv_gDF/view)
- 🟢 **Negative (Normal) Samples**: [Download](https://drive.google.com/file/d/18QJyRNVDG6jguNmV04GRuZM98IGdizUb/view)

Each dataset contains:
- `.png` images representing X-ray scans
- `.json` files containing related metadata (if available)

---

## 🧠 Model Architecture

- **Architecture**: Siamese Neural Network
- **Backbone**: Pre-trained ResNet-18
- **Loss Function**: Contrastive Loss
- **Similarity Metric**: Euclidean Distance

The model learns to compute feature embeddings and compare image similarity to detect anomalies.

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

## 🚀 How to Run

1. **Clone this repository**
   ```bash
   git clone https://github.com/anamikapati/xray-anomaly-detection.git
   cd xray-anomaly-detection
