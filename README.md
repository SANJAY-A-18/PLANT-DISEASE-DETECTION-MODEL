# 🌿 Plant Disease Detection with EfficientNetB0 (Keras + TensorFlow)

This project implements a **deep learning-based image classifier** that identifies plant diseases from leaf images. It uses **EfficientNetB0** as the backbone and is trained on an augmented dataset using **transfer learning**, **mixed precision**, and **checkpointing** to ensure high accuracy and optimized training time.

---

## 🧠 Project Summary

This system enables you to:
- Preprocess and augment plant disease datasets
- Train a deep CNN model with transfer learning (EfficientNetB0)
- Resume training from checkpoints
- Evaluate and visualize model performance
- Classify plant disease from test images

---

## 🚀 Features

- 🌾 Supports 38+ plant disease categories
- 🧪 Augmentation for robust training
- 🧠 EfficientNetB0 as base model with fine-tuning
- 🪢 Checkpoint saving and resuming
- 📈 Confusion matrix and classification report
- 📊 Accuracy & loss plots for training history
- 🖼️ Predicts plant disease from test images with filenames

---

## 🧱 Project Structure

```bash
├── TRAINING.py              # Model training with EfficientNet and callbacks
├── PREDICTION.py            # Loads model and predicts test image classes
├── epoch_checkpoints/       # Stores weights for each epoch
├── FARM AI/
│   ├── train/               # Augmented training images
│   ├── valid/               # Validation images
│   └── test/                # Test images for final prediction
