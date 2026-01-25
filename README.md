# CIFAR-10 Image Classification: A Transfer Learning Approach with ResNet50
![Sample Predictions](./reports/sample_predictions.png)

## 📌 Project Overview
Transitioning from the grayscale simplicity of Fashion MNIST, this project tackles the **CIFAR-10** dataset—a benchmark in computer vision consisting of 60,000 $32 \times 32$ color images across 10 mutually exclusive classes.

The primary objective was to leverage **Transfer Learning** using the **ResNet50** architecture, demonstrating the efficiency of using pre-trained weights (ImageNet) to solve complex image classification tasks even with relatively low-resolution inputs.

---

## 📊 Dataset Insights
CIFAR-10 presents a unique challenge due to the low resolution of the images and the specific nuances of its classes:

* **Total Images:** 60,000 (50k Training / 10k Testing).
* **Dimensions:** $32 \times 32$ RGB.
* **Classes:** ✈️ Airplane, 🚗 Automobile, 🐦 Bird, 🐱 Cat, 🦌 Deer, 🐶 Dog, 🐸 Frog, 🐴 Horse, 🚢 Ship, 🚛 Truck.
* **Nuance:** The "Automobile" class includes sedans and SUVs, while "Truck" is reserved for heavy-duty vehicles. Neither includes pickup trucks, requiring the model to learn subtle boundary features.

---

## ⚙️ Technical Workflow

### 1. Image Preprocessing
Before feeding the images into the network, the following steps were implemented:
* **Normalization:** Scaling pixel values to a $[0, 1]$ or $[-1, 1]$ range depending on the ResNet50 requirements.
* **Label Encoding:** One-hot encoding the 10 categorical labels.
* **Input Shape:** Fixed at $(32, 32, 3)$ to match the dataset without adding unnecessary computational overhead through resizing.

### 2. Model Architecture
The model utilizes a **Transfer Learning** strategy divided into two main components:
* **The Base Model:** ResNet50 with `weights='imagenet'`. The top layer was removed (`include_top=False`) to allow for custom classification.
* **The Custom Head:** A sequence of dense layers designed to map the ResNet features to the 10 CIFAR classes.
    * `GlobalAveragePooling2D` to reduce dimensionality.
    * `Dense Layer` (128 neurons, ReLU).
    * `Dense Layer` (64 neurons, ReLU).
    * `Output Layer` (10 neurons, Softmax). 

---

## 🚀 Training Strategy
The training was executed in two distinct phases to ensure stability and maximize accuracy:

| Phase | Description | Epochs | Rationale |
| :--- | :--- | :--- | :--- |
| **Phase 1: Feature Extraction** | Base layers frozen. Only the custom head is trained. | 10 | Prevents the large gradients of the random head from "destroying" the pre-trained ImageNet weights. |
| **Phase 2: Fine-Tuning** | Base layers unfrozen. The entire network is trained at a low learning rate. | 10 | Allows the model to adapt the specialized filters in the ResNet base to the specific textures and shapes of CIFAR-10. |

---

## 🧠 Analytical Reasoning & Observations

### Why Transfer Learning?
Training a deep residual network from scratch on $32 \times 32$ images often leads to overfitting or convergence issues. By starting with **ResNet50** pre-trained on ImageNet, we "inherit" the ability to detect edges, textures, and patterns, which drastically speeds up convergence.

### Observations During Training
* **The "Head" Training:** Initially, the loss decreased rapidly, but accuracy peaked quickly as the base remained static.
* **Fine-Tuning Impact:** Unfreezing the model led to a significant jump in validation accuracy, though it increased the risk of overfitting due to the high parameter count relative to the image size.
* **Hardware Constraints:** Training was conducted on **Google Colab** using a GPU runtime. Without GPU acceleration, the fine-tuning phase would be computationally prohibitive for a standard CPU.

---

## 🏁 Conclusion & Future Improvements
While the current model provides a solid baseline, further improvements could include:

* 🖼️ **Data Augmentation:** Introducing rotations and horizontal flips to improve generalization.
* 📉 **Learning Rate Schedulers:** Implementing a decay to stabilize the fine-tuning phase.
* ⏳ **Extended Training:** Increasing epochs beyond 10+10 to reach the model's full potential.
