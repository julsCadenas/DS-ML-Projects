# 1️⃣ MNIST Handwritten Digit Classifier

A simple neural network built with TensorFlow/Keras to classify handwritten digits (0–9) from the MNIST dataset.

## 📊 Dataset
- **Dataset:** MNIST (from `tf.keras.datasets`)
- **Image shape:** 28x28 grayscale
- **Train samples:** 60,000
- **Test samples:** 10,000

## 🧠 Model Architecture
```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

## ✅ Evaluation Results
- **Test Accuracy:** 97.69%
- **Test Loss:** 0.0917