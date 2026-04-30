# Autoencoder for Image Compression (PyTorch)

## 🧠 Overview
This project implements a simple **Autoencoder** using PyTorch to compress and reconstruct images from the MNIST dataset.

The model learns a lower-dimensional representation (latent space) of images and reconstructs them with minimal loss.

---

## 🚀 Features
- Fully connected Autoencoder
- Training on MNIST dataset
- Image reconstruction visualization
- Lightweight and easy to extend

---

## 🏗️ Architecture
- Encoder: 784 → 128 → 64 → 16
- Decoder: 16 → 64 → 128 → 784

---

## 📦 Setup (Poetry)

Install Poetry:
```bash
pip install poetry