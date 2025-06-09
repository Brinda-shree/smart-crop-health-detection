# ⚙️ Optimization Techniques Used

### ✅ Transfer Learning
- Used pre-trained MobileNetV2.
- Avoided training large number of parameters from scratch.

### ✅ Data Augmentation
- Helps avoid overfitting by generating diverse images.
- Rotation, flipping, zooming.

### ✅ EarlyStopping
- Monitored `val_loss` and stopped training early if it stopped improving.

### ✅ Dropout
- Added Dropout (0.3) before final layers to regularize.

### 📈 Result
- Achieved validation accuracy of 94.7% (example).
- Prevented overfitting between training and validation loss curves.
