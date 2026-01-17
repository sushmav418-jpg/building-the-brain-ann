# building-the-brain-ann
ANN and deep learning fundamentals using TensorFlow
# 🧠 Building the Brain using Artificial Neural Networks (ANN)

This project demonstrates the fundamental concepts of **Artificial Neural Networks (ANN)** by building, training, and evaluating a simple neural network using **TensorFlow and Keras**. The goal is to understand how machines learn from data and make decisions, similar to how a human brain works.

---

## 📌 Project Overview

Artificial Neural Networks are inspired by the structure and functioning of the human brain. In this project, we:
- Convert image data into numerical form
- Pass data through artificial neurons
- Train the model using a loss function and optimizer
- Evaluate model performance using accuracy metrics

The project uses the **Fashion-MNIST dataset**, which consists of grayscale images of clothing items.

---

## 🧠 Concepts Covered

- Artificial Neurons and Weights
- Linear Models (y = wx + b)
- Multivariate Inputs
- Image Representation as Pixels
- ANN Architecture
- Loss Functions
- Optimizers
- Model Training and Evaluation

---

## 📂 Dataset Used

### Fashion-MNIST
- 70,000 grayscale images (28×28 pixels)
- 10 different clothing categories
- 60,000 training images
- 10,000 validation images

**Classes include:**
- T-shirt / Top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

---

## 🏗️ Model Architecture

The neural network consists of:

1. **Flatten Layer**
   - Converts 28×28 images into a 1D array of 784 values

2. **Dense Layer**
   - Fully connected layer with 10 neurons
   - Each neuron represents one clothing category

---

## ⚙️ Technologies Used

- Python
- TensorFlow
- Keras (tf.keras)
- Jupyter Notebook
- NumPy

---

## 🧪 Loss Function & Metrics

### Loss Function
- **Sparse Categorical Cross-Entropy**
  - Suitable for multi-class classification
  - Works with integer labels
  - Penalizes confident wrong predictions

### Metrics
- **Accuracy**
  - Measures how many predictions are correct

---

## 🚀 How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/sushmav418-jpg/building-the-brain-ann.git

