# Deep Learning Benchmarking Suite: Tabular, Vision, Sequential, and Graph Data

This repository contains a comprehensive suite of **4 Jupyter Notebooks** dedicated to exploring, preprocessing, and modeling diverse data modalities using Machine Learning and Deep Learning techniques. Each notebook contains end-to-end pipelines including Exploratory Data Analysis (EDA), advanced preprocessing, class imbalance handling, and thorough evaluation metrics.

## 📊 Repository Overview

Every notebook includes detailed performance evaluations accompanied by:
- **Training History Curves** (Loss and Accuracy/Metric plots over epochs)
- **Confusion Matrices** for multi-class error analysis
- **ROC/AUC Curves** for class-discrimination performance

---

## 📓 Notebook 1: Tabular Data Classification
**Dataset:** Forest Cover Type  
**Task:** Multi-class classification of forest cover types based on cartographic variables.

### 🛠️ Pipeline & Features
- **EDA & Preprocessing:** Feature engineering, scaling, and correlation analysis.
- **Imbalance Mitigation:** Applied advanced resampling techniques to fix severe class distribution skews.
- **Models Evaluated:** Logistic Regression, Random Forest, XGBoost, and custom Deep Neural Networks (DNNs).

### 🏆 Results Summary
| Model | Accuracy |
| :--- | :---: |
| **Random Forest** | **95.00%** 🥇 |
| **Custom Neural Network (DNN)** | **92.00%** 🥈 |
| **XGBoost** | **91.00%** |

---

## 📓 Notebook 2: Computer Vision & Explainable AI (XAI)
**Dataset:** Intel Image Classification  
**Task:** Image scene classification across 6 natural categories.

### 🛠️ Pipeline & Features
- **Preprocessing:** Image resizing, normalization, and dynamic **Data Augmentation** techniques to combat overfitting.
- **Architectures:** Built custom Deep Convolutional Neural Networks (DCNN) and utilized **Transfer Learning** across 6 state-of-the-art backbones.
- **Explainable AI (XAI):** Implemented **Grad-CAM** and filter analysis to visualize convolutional feature maps and model focus areas.

### 🏆 Results Summary
- **Best Performing Model:** Transfer Learning DCNN 
- **Top Accuracy:** **89.50%**
- **Evaluated Architectures:** VGG16, VGG19, ResNet50, InceptionV3, MobileNetV2, EfficientNetB0.

---

## 📓 Notebook 3: Sequential & Time-Series Data
**Dataset:** Electric Devices  
**Task:** Sequence classification of electricity consumption profiles.

### 🛠️ Pipeline & Features
- **EDA:** Sequence length verification and temporal pattern visualization.
- **Class Imbalance:** Integrated custom class weighting and resampling methods to address structural dataset imbalance.
- **Architectures:** Scaled from simple recurrent baselines to advanced architectures including SimpleRNN, GRU, LSTM, Conv1D, Bidirectional layers, hybrid/mixed configurations, and Transformer Encoders.

### 🏆 Results Summary
- **Best Performing Model:** Hybrid **LSTM + Conv1D** Architecture
- **Top Accuracy:** **89.00%**
- **Key Insight:** Merging temporal representations (LSTM) with localized spatial filters (Conv1D) yielded the most robust sequence boundaries.

---

## 📓 Notebook 4: Graph Neural Networks (GNN)
**Dataset:** ogbn-arxiv  
**Task:** Node classification in a citation network (predicting the subject area of arXiv papers).

### 🛠️ Pipeline & Features
- **EDA:** Graph topology analysis, degree distributions, and homophily measurement.
- **Baselines:** Flattened features processed via Random Forest, XGBoost, and Multi-Layer Perceptrons (MLP).
- **Graph Models:** Evaluated baseline and deeply optimized variants of Graph Convolutional Networks (GCN), GraphSAGE, Graph Attention Networks (GAT), and Graph Neural Networks with recurrent cells (GRNN).

### 🏆 Results Summary
- **Best Performing Model:** **GraphSAGE** (Advanced Variant)
- **Top Accuracy:** **70.00%**
- **Key Insight:** Neighborhood sampling and aggregation techniques (GraphSAGE) generalized significantly better to unseen citation structures compared to standard isotropic convolutions.

---

## 📈 Evaluation Artifacts Inside

Each notebook contains generated plots ready for review:
1. **`history_plots`**: Epoch-by-epoch tracking of training/validation metrics.
2. **`confusion_matrices`**: Visual grids highlighting precisely where models misclassified instances.
3. **`roc_curves`**: Multi-class One-vs-Rest ROC curves detailing AUC values per class label.

## ⚙️ Requirements & Getting Started

### Prerequisites
- Python 3.9+
- Jupyter Lab / Notebook
- Core Libraries: `torch` / `tensorflow`, `torch_geometric` (for GNNs), `sklearn`, `xgboost`, `cv2` (Grad-CAM), `matplotlib`, `seaborn`.

### Setup
```bash
# Clone the repository
git clone [https://github.com/pavlvs-91/ggsn_projects.git](https://github.com/pavlvs-91/ggsn_projects.git)
cd ggsn_projects
```

### Launch notebooks
```bash
jupyter notebook
```
