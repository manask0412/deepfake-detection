# **Deepfake Detection**
A Deep Learning model designed to classify images as Real or AI-generated (Fake). Built with TensorFlow, evaluated using the CIFAKE dataset and deployed via a Gradio app for interactive use.

## 📚 Project Overview
This project tackles the challenge of distinguishing between AI-generated (fake) images and real images. In response to the growing sophistication of generative models (e.g., Adobe Firefly, Stable Diffusion, Midjourney), we build a binary classification model that leverages a pre-trained EfficientNetB2 network to effectively classify images as **Real** or **Fake**. Our approach emphasizes robust performance even under adversarial perturbations.

## 🔗 Live Demo
[![Live Demo](https://img.shields.io/badge/Live%20Demo-On%20HuggingFace-blue?style=flat-square&logo=huggingface)](https://huggingface.co/spaces/manas0412/deepfake-detection)

## 🧰 Tech Stack
- **Core Frameworks:** Python 3.11.11, TensorFlow 2.18, Keras  
- **Adversarial Robustness:** ART (Adversarial Robustness Toolbox)  
- **Data Handling & Visualization:** NumPy, scikit-learn, Matplotlib, Seaborn  
- **Demo Interface:** Gradio (deployed on HuggingFace Spaces)

## 🗃️ Dataset
- **Source:** [CIFAKE (Kaggle)](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)  
- **Train:** 50,000 Real, 50,000 Fake (32×32)  
- **Test:** 10,000 Real, 10,000 Fake (32×32)

## 🧠 Model Architecture
The model is built on **EfficientNetB2** with **ImageNet** pre-trained weights (frozen initially). The architecture is as follows:
1. **Base Model**: EfficientNetB2 (without the top classification layers). 
2. **GlobalAveragePooling2D**: Reduces spatial dimensions. 
3. **Dropout (0.4)**: Regularization technique to prevent overfitting.
4. **Dense Output Layer**: Single neuron with a sigmoid activation to produce a binary classification (**Real** or **Fake**).

## 🚀 Training Strategy
1. **Initial Training Phase (20 epochs)**  
   - Freeze EfficientNetB2 base weights.  
   - Optimizer: Adam (learning rate = 1e-3).  
   - Label smoothing applied (0.1) to improve generalization.

2. **Adversarial Attack and Training (5 epochs)**  
   - Generate **Adversarial Examples** using **FGSM** attack (ε = 0.05) on 30% of the training set.  
   - Train on a mixture of clean and adversarial examples to enhance robustness.

3. **Fine-Tuning Phase (10 epochs)**  
   - Unfreeze the entire EfficientNetB2 backbone.  
   - Optimizer: Adam (learning rate = 1e-5) for gentle weight updates.

4. **Threshold Optimization**  
   - Find the best classification threshold by analyzing the Precision-Recall (PR) curve on the validation set.

5. **Training Callbacks:**  
    - `EarlyStopping` Stops training early if Validation Loss does not improve, to prevent overfitting and save compute time.  
    - `ReduceLROnPlateau` Reduces Learning Rate when the validation loss plateaus for smoother convergence.

## 🧪 Evaluation
The model is evaluated across three scenarios:
- **Standard Test**: Evaluation on the original, clean (unaltered) test dataset.
- **Adversarial Robustness Test**: Evaluation on adversarial examples generated via FGSM (ε = 0.05) for 30% of the test samples.
- **Mixed Robustness Test**: Evaluation on a combined set of clean and adversarial images.

### Metrics Used:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC Score
  - Confusion Matrix (with visualization)
  - Inference Latency (average, median, 95th percentile)

## 📊 Test Results
| Test Data                  | Accuracy  | Precision | Recall | F1 Score | Misclassifications | Latency (ms) | AUC |
|-----------------------------|-----------|-----------|--------|----------|--------------------|--------------|---------|
| **Original Images**         | 94.72%    | 94.12%    | 95.40% | 94.76%   | 1056/20000 (5.28%)  | 0.31         | 0.9872    |
| **FGSM Adversarial Images**         | 94.65%    | 94.06%    | 95.40% | 94.73%   | 321/6000 (5.35%)  | 0.32         | 0.9870    |
| **(Original + Adversarial) Images**  | 94.70%    | 94.11%    | 95.40% | 94.75%   | 1377/26000 (5.30%)  | 0.31         | 0.9871    |

### Key Insights:
- **Standard Test:** Excellent performance with high accuracy, precision, recall, and F1 score. Low misclassification rate. ✅
- **Adversarial Robustness Test:** Minimal performance drop under adversarial conditions, maintaining robustness. 🔥
- **Mixed Robustness Test:** Consistent performance on both clean and adversarial datasets, demonstrating strong generalization. 💪

