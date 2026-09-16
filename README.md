# 🤖 Machine Learning & AI Codebase (`ml-learning`)

> Comprehensive, hands-on repository covering **Supervised Learning**, **Unsupervised Learning**, **Reinforcement Learning (Q-Learning)**, **Computer Vision (OpenCV & YOLOv8)**, **PyTorch Deep Learning**, **Pandas Data Science**, and **TensorFlow Fundamentals**.

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

---

## 📌 Repository Modules & Highlights

### 1. 📈 Supervised Learning (`supervised_practice/` & `revision/`)
- **Regression Models**: Simple Linear Regression, Multiple Linear Regression, Polynomial Regression, Ridge, Lasso, and ElasticNet Regularization.
- **Classification Models**: Logistic Regression, Decision Trees, K-Nearest Neighbors (KNN), Naive Bayes, Support Vector Machines (SVM).
- **Real-World Case Studies**:
  - `disease_prediction.py`: Multiclass health diagnostic predictor.
  - `mail_detection.py`: Spam vs. Ham email classifier.
  - `temperature_forecasting.py`: Time-series temperature predictor.
  - `house_price_prediction.py`: Real estate valuation pipeline.

### 2. 🧩 Unsupervised Learning (`unsupervised_practice/` & `unsupervised/`)
- **Clustering Algorithms**:
  - K-Means Clustering & Elbow Method for optimal cluster evaluation ($k$).
  - Agglomerative Hierarchical Clustering with Dendrogram analysis.
  - DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
- **Dimensionality Reduction**:
  - Principal Component Analysis (PCA) & Linear Discriminant Analysis (LDA).
- **Association Rule Learning**:
  - Apriori algorithm for market basket analysis.

### 3. 🎮 Reinforcement Learning (`reinforcement_learning/`)
- **Q-Learning Algorithm**: Tabular Q-learning with Epsilon-Greedy exploration/exploitation decay.
- **Environment Implementations**:
  - GridWorld Navigation: Agent finding optimal path to bottom-right destination.
  - Traffic Light Signal Optimization: Q-value based signal switching to reduce congestion.

### 4. 👁️ OpenCV & Object Detection (`opencv/`)
- **OpenCV Basics**: Image transformations, color spaces, blurring, thresholding, edge detection, and contour analysis.
- **YOLOv8 & Object Tracking**:
  - `01-yolo_basics.py`: YOLOv8 object detection on static images.
  - `02-yolo_webcam.py`: Real-time camera feed object detection.
  - `04-person_tracking.py` & `06-bytetrack_detection.py`: Multi-object tracking with ByteTrack.
  - `09-entry_exit_counter.py`: Line-crossing entry & exit analytics counter.

### 5. 🧠 PyTorch Deep Learning (`pytorch_01/` & `pytorch02/`)
- **Neural Network Architecture**: Custom `nn.Module` classes, linear layers, activation functions (ReLU, Sigmoid, Softmax).
- **Convolutional Neural Networks (CNNs)**:
  - Custom CNNs, ResNet architectures, Residual Blocks (Skip Connections), Stride & Padding mechanics.
  - MNIST Handwritten Digit Classification.

---

## 📁 Directory Structure

```
ml-learning/
├── supervised_practice/       # Linear/Logistic Regression, Decision Trees, KNN, SVM
├── unsupervised_practice/     # K-Means, Elbow Method, PCA, LDA, DBSCAN, Apriori
├── reinforcement_learning/    # Q-Learning, Epsilon-Greedy, Traffic & GridWorld
├── opencv/                    # OpenCV Filters, YOLOv8 Detection, ByteTrack Counter
├── pytorch_01/                # PyTorch Tensors, Autograd, Convolutions
├── pytorch02/                 # ResNet, Residual Blocks, MNIST CNN Classifier
├── pandas_basics/             # Data Frames, Missing Value Imputation, Selection
└── tensorflow/                # Neural Network Fundamentals
```

---

## 🛠️ Environment Setup & Installation

```bash
# Clone the repository
git clone https://github.com/dipesh097/ml-learning.git
cd ml-learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core packages
pip install numpy pandas matplotlib scikit-learn opencv-python torch torchvision ultralytics
```

Run any script:
```bash
python supervised_practice/"Disease Prediction.py"
```
