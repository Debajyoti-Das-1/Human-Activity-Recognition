# 🧠 Human Activity Recognition (HAR)
### 🚀 Parallel Dual-Stream Hybrid System for Time-Series Intelligence

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch"/>
  <img src="https://img.shields.io/badge/Hardware-M4_Pro_Optimized-success?style=for-the-badge&logo=apple"/>
  <img src="https://img.shields.io/badge/Captum-Axiomatic_XAI-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Accuracy-91.62%25-green?style=for-the-badge"/>
</p>

---

## 📌 Overview

This repository implements a **State-of-the-Art (SOTA) Human Activity Recognition system** using high-frequency (**50Hz**) multivariate sensor data.

The core contribution is a **Parallel Dual-Stream Hybrid architecture** that fuses:

- 🔁 **Bidirectional Temporal Dynamics** (Bi-LSTM)  
- 🌐 **Global Spatial Attention** (Transformer)  

This design enables the model to **simultaneously capture motion flow and posture context**, achieving:

- 🚀 **91.62% accuracy**
- ⚡ Optimized inference on **Apple Silicon (MPS / M4 Pro)**
- 🧠 **Physically interpretable decision-making**

> 💡 This is not just a model — it is a **research-to-production ML system**.

---

## 🧠 Architecture: Parallel Advanced Hybrid

Instead of naive sequential stacking, the model uses a **Shared CNN Stem + Parallel Dual Streams** to explicitly solve the:

> ⚖️ **Temporal–Spatial Trade-off**

<p align="center">
  <img src="images/model_architecture.jpg" width="850"/>
</p>

---

## ⚙️ Technical Breakdown

### 🔹 Shared CNN Stem (Feature Compression)

- 1D-CNN extracts **local motion signatures** (e.g., foot-strike patterns)
- **MaxPool1d (128 → 64)** reduces temporal resolution  
- Acts as a **noise filter + computational bottleneck optimizer**

---

### 🔹 Stream A: Temporal Dynamics (Bi-LSTM)

- Captures **chronological motion flow**
- Bidirectional processing:
  - Forward: $\overrightarrow{h}_t$
  - Backward: $\overleftarrow{h}_t$
- Learns directional patterns:
  - ✅ Walking Upstairs vs Downstairs

---

### 🔹 Stream B: Global Context (Transformer)

- Uses **4-head Self-Attention**
- Observes the **entire 2.56s window simultaneously**
- Overcomes:
  - ❌ Sequential memory limitations  
  - ❌ Information decay  

Learns:
- Posture distribution  
- Gravity alignment  
- Static activity cues  

---

### 🔹 Fusion & Decision Head

- Concatenation: