##  Edge AI Material Classifier

A lightweight CNN model for classifying recyclable materials — **plastic**, **paper**, and **glass** — using stylized digit-like images. Built for edge deployment with TensorFlow and quantization-ready via `tensorflow_model_optimization`.

---

### Project Overview

This project demonstrates a full edge AI pipeline:

- Synthetic dataset creation using MNIST digits remapped to material classes
- Preprocessing: resizing, RGB conversion, normalization
- Lightweight CNN architecture optimized for edge devices
- One-hot label encoding and categorical training
- Visualization of predictions and training metrics
- Ready for quantization and `.tflite` export

---

### Model Architecture

```python
Conv2D(32) → BatchNorm → MaxPool  
Conv2D(64) → BatchNorm → MaxPool  
Conv2D(64) → BatchNorm  
Flatten → Dropout(0.5) → Dense(64) → BatchNorm → Dropout(0.3)  
Dense(3, softmax)
```

Designed for low-latency inference and small footprint.

---

### Dataset

- Source: MNIST digits
- Classes: `['plastic', 'paper', 'glass']`
- Mapping: `digit % 3` to simulate material categories
- Preprocessing:
  - Resized to `(32, 32)`
  - Converted to RGB
  - Normalized to `[0, 1]`
  - One-hot encoded labels

---

### Training

```python
model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=5,
    validation_data=(x_test, y_test),
    verbose=1
)
```

Training history is stored for visualization and evaluation.

---

### Visualization

- Sample predictions with class labels
- Training accuracy and loss curves (optional)
- Confusion matrix (optional)

---


### Requirements

```bash
tensorflow-macos==2.13.0  
tensorflow-metal==1.1.0  
tensorflow-model-optimization  
matplotlib  
numpy
```

---

