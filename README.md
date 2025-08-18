# U-Net-variants-for-predicting-SBP-and-DBP-from-PPG-signals  

This repository contains deep learning models developed for predicting **Systolic Blood Pressure (SBP)** and **Diastolic Blood Pressure (DBP)** from **Photoplethysmogram (PPG) signals**.  
The work explores multiple **U-Net variants** with different activation functions to optimize prediction performance.  

## 🎯 Objective  
The goal is to build robust deep learning models that can accurately **predict SBP and DBP values** from preprocessed PPG signals and their derivatives.  

---

## 📌 Models Implemented  
- **U-Net** – Standard U-Net model (largest parameter size).  
- **Squeeze U-Net** – Lightweight version using squeeze-and-expand layers.  
- **Mobile U-Net** – Efficient architecture inspired by MobileNet blocks.  
- **Half U-Net** – Reduced-depth U-Net with residual connections (smallest size).  

---

## ⚡ Activation Functions Explored  
- **ReLU**  
- **Swish**  
- **Mish**  

These were tested across different architectures for performance comparison.  

---

## Data Processing  
- **Input Signal:** PPG sampled at **125 Hz**, duration **10 seconds**.  
- **Preprocessing:**  
  - Bandpass filter (**0.2 – 8 Hz**) to remove **baseline wandering** and **high-frequency noise**.  
- **Feature Extraction:**  
  - Raw PPG  
  - First derivative (velocity)  
  - Second derivative (acceleration)  
- **Windowing:**  
  - Signal divided into **512-point windows** (to fit U-Net max pooling stages).  
  - **50% overlap** between consecutive windows to ensure continuity.  

---



## 📂 Repository Structure  
