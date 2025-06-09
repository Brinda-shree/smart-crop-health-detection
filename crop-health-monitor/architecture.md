# 🧠 Model Architecture Report

## Model Used: MobileNetV2 (Transfer Learning)

### ✅ Why MobileNetV2?
- Lightweight CNN model trained on ImageNet
- Fast and efficient for embedded systems/agriculture edge devices
- Performs well with limited data

### 🧱 Architecture Overview
- Input: 128x128 RGB images
- Base: MobileNetV2 (frozen layers)
- Custom Head:
  - GlobalAveragePooling2D
  - Dropout (0.3)
  - Dense (128 neurons, ReLU)
  - Output: Dense Softmax (3 classes)

### 📦 Classes
- `Potato___Early_blight`
- `Tomato_healthy`
- `Tomato_Late_blight`

### 🔧 Image Augmentation
- Rotation: 20°
- Zoom: 0.2
- Horizontal Flip
- Rescaled to [0, 1]
