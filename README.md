# GraduationProject: Neural Style Transfer Platform for Real-Time Image and Video Stylization

## Overview

This project was developed as my final-year project for the BSc Computer Science programme at Goldsmiths, University of London, where it received a **Distinction** grade.

The project investigates the trade-offs between image quality, style fidelity, flexibility, and computational efficiency across several state-of-the-art Neural Style Transfer approaches

The project goes beyond traditional Neural Style Transfer implementations by introducing enhanced architectures, multi-style transfer capabilities, and video stylization with temporal consistency.

---

## Objectives

The primary goals of the project were:

* Investigate the strengths and limitations of existing Neural Style Transfer methods.
* Improve the scalability and efficiency of NST for real-time applications.
* Compare multiple NST architectures using qualitative and quantitative evaluation metrics.
* Extend style transfer capabilities to support:

  * Arbitrary style transfer
  * Multi-style transfer
  * Real-time video style transfer
---

## Implemented Models

### 1. Gatys et al. (2016)

The original optimization-based Neural Style Transfer approach using VGG-19 feature extraction and style/content loss optimization.

### 2. Johnson et al. (2016)

A feed-forward neural network for real-time image stylization.

### 3. AdaIN (Adaptive Instance Normalization)

An arbitrary style transfer model capable of applying unseen artistic styles without retraining.

### 4. Dumoulin et al. (2017)

Conditional Instance Normalization (CIN) model supporting multiple artistic styles.

### 5. Dumoulin V2 (Proposed Extension)

An enhanced architecture combining CIN and AdaIN techniques to improve style flexibility and generalization to unseen styles.

### 6. Dumoulin V2 Multi-Style

An extension capable of applying multiple artistic styles simultaneously within a single image.

### 7. Dumoulin V2 Video Transfer

A video style transfer system using optical flow to maintain temporal consistency and reduce flickering artifacts.

---

## Technologies Used

### Programming Languages

* Python

### Deep Learning Frameworks

* PyTorch
* TensorFlow
* Keras

### Computer Vision & Image Processing

* OpenCV
* Pillow (PIL)
* ImageIO

### Data Analysis & Visualization

* NumPy
* Matplotlib
* Seaborn

### Deployment & Development

* Streamlit
* GitHub
* Google Colab
* FFmpeg

---

## Evaluation Metrics

The implemented models were evaluated using:

* Content Loss
* Style Loss
* SSIM (Structural Similarity Index)
* LPIPS (Learned Perceptual Image Patch Similarity)
* Processing Time
* User Surveys and Qualitative Evaluation

---

## Results

The project demonstrated the trade-offs between image quality, flexibility, and computational efficiency across different NST architectures.

Key findings include:

* Gatys et al. produced the highest style fidelity but required significant computation time.
* Johnson et al. enabled real-time stylization through feed-forward networks.
* AdaIN provided fast arbitrary style transfer without retraining.
* Dumoulin-based architectures achieved improved flexibility with multiple styles.
* The proposed Dumoulin V2 extensions successfully supported unseen styles, multi-style transfer, and video stylization while maintaining high visual quality.

---

## Web Application

A Streamlit-based web application was developed to allow users to upload content and style images, experiment with different NST models, and generate stylized outputs in real time.

Live Demo:
https://neural-style-transfer-graduation-25.streamlit.app/


