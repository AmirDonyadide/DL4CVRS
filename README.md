# 🚗 Car vs. Not Car Classifier

This project builds and evaluates two image classification models to distinguish between **car** and **not_car** images. It includes a custom Convolutional Neural Network (CNN) and a transfer learning model based on ResNet18.

---

## 📁 Project Structure

```
.
├── data/
│   ├── train/
│   │   ├── car/
│   │   └── not_car/
│   └── test/
│       ├── car/
│       └── not_car/
├── Assignment1.ipynb             # Full training, evaluation, and visualization script
├── requirements.txt              # List of required Python packages
└── README.md                     # Project documentation
```

---

## 🧠 Models Used

- **Custom CNN**: 1 conv + pooling + dropout + 2 fully connected layers
- **Pretrained ResNet18**: fine-tuned with frozen feature layers

---

## 📊 Features

- Dataset normalization (mean and std computation)
- Separate training for both models
- TensorBoard logging for loss tracking
- Evaluation on both **train** and **test** sets:
  - Accuracy
  - Classification report
  - Confusion matrix
- Visual display of 8 misclassification groups (car → not_car, not_car → car)

---

## 🚀 How to Run

1. Prepare the dataset under `data/train/` and `data/test/` with `car/` and `not_car/` subfolders.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```bash
   python Assignment1.ipynb
   ```
4. Launch TensorBoard (optional):
   ```bash
   tensorboard --logdir=runs
   ```

---

## 📦 Requirements

- torch
- torchvision
- numpy
- matplotlib
- scikit-learn
- tensorboard

(See `requirements.txt` for versions)

---

## 👨‍💻 Author

Group 5
Course Deep Learning for Computer Vision and Remote Sensing
KIT