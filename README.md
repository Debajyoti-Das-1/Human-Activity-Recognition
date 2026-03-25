# 🧠 Human Activity Recognition (HAR)
### 🚀 Dual-Stream Deep Learning System for Time-Series Intelligence

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-DeepLearning-EE4C2C?style=for-the-badge&logo=pytorch"/>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/Captum-ExplainableAI-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Research--to--Production-success?style=for-the-badge"/>
</p>

---

## 📌 TL;DR

A **research-grade Human Activity Recognition system** that:

- ⚡ Processes **50Hz multivariate sensor data**
- 🧠 Benchmarks CNN, LSTM, Transformer architectures
- 🚀 Proposes a **novel Dual-Stream Hybrid model**
- 🔍 Uses **Explainable AI (Captum)** to interpret decisions
- 📡 Simulates **real-time inference pipelines**

> 💡 Designed like a real-world ML system — not just a notebook experiment.

---

## 🎯 Problem Statement

Human Activity Recognition (HAR) is fundamentally challenging due to:

> ⚖️ **Temporal–Spatial Trade-off**

- Short-term → motion signatures (e.g., steps, transitions)  
- Long-term → posture & context (e.g., sitting vs standing)

Most models fail to capture both **simultaneously**.

---

## 🧠 Solution: Dual-Stream Hybrid Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/yourusername/assets/main/har_architecture.png" width="700"/>
</p>

### ⚙️ Architecture Breakdown

| Component | Role |
|----------|------|
| **CNN Stem** | Noise filtering + feature compression |
| **Bi-LSTM Stream** | Temporal dynamics (forward + backward) |
| **Transformer Stream** | Global self-attention (posture understanding) |
| **Fusion Head** | Combines representations for final prediction |

---

## 📊 Model Benchmarking

| Model | Accuracy | Macro F1 | Insight |
|------|---------|----------|--------|
| **1D-CNN** | 90.91% | 0.9100 | Fast, but noise-sensitive |
| **LSTM** | 90.74% | 0.9078 | Good sequence modeling, unstable |
| **CNN-LSTM** | 90.97% | 0.9114 | Strong baseline |
| **Transformer** | 88.94% | 0.8883 | Stable, global context |
| **🚀 Advanced Hybrid** | **91.62%** | **0.9167** | **Best performance** |

---

## 🔬 Explainability (XAI)

Using **Captum + Integrated Gradients**:

### 🧩 Key Insight: *Sitting vs Standing*

- Accelerometer signals ≈ identical  
- Traditional models fail ❌  

### ✅ Model Discovery:

- Ignores accelerometer  
- Focuses on **Body Gyro X**

> 💡 The model learns **micro-rotational sway (inverted pendulum effect)** — a real physical phenomenon.

👉 This confirms the model is learning **physics-aware representations**, not just correlations.

---

## 🚀 Quick Start

### 1️⃣ Clone & Install

```bash
git clone https://github.com/Debajyoti-Das-1/Human-Activity-Recognition.git
cd har_project
pip install -r requirements.txt