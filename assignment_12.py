# Install dependencies (if not already)
#!pip install tensorflow matplotlib

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# -------------------------------
# 1️⃣ Load MNIST dataset
# -------------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape for CNN: (batch, height, width, channels)
x_train = x_train.reshape(-1,28,28,1).astype('float32') / 255.0
x_test = x_test.reshape(-1,28,28,1).astype('float32') / 255.0

y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

# -------------------------------
# 2️⃣ Define a simple CNN
# -------------------------------
def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -------------------------------
# 3️⃣ Training without augmentation
# -------------------------------
cnn_plain = create_cnn_model()
history_plain = cnn_plain.fit(
    x_train, y_train_cat,
    validation_split=0.1,
    epochs=5,
    batch_size=128
)

# -------------------------------
# 4️⃣ Training with data augmentation
# -------------------------------
# Define different augmentation techniques
datagen = ImageDataGenerator(
    rotation_range=15,      # rotate images randomly by ±15 degrees
    width_shift_range=0.1,  # shift horizontally by ±10%
    height_shift_range=0.1, # shift vertically by ±10%
    zoom_range=0.1,         # zoom in/out by ±10%
    shear_range=0.1,        # shear transformation
    horizontal_flip=False   # MNIST digits shouldn't be flipped horizontally
)

datagen.fit(x_train)

cnn_aug = create_cnn_model()

# Train using augmented data
history_aug = cnn_aug.fit(
    datagen.flow(x_train, y_train_cat, batch_size=128),
    validation_split=0.1,
    epochs=5,
    steps_per_epoch=len(x_train)//128
)

# -------------------------------
# 5️⃣ Evaluate models
# -------------------------------
loss_plain, acc_plain = cnn_plain.evaluate(x_test, y_test_cat)
loss_aug, acc_aug = cnn_aug.evaluate(x_test, y_test_cat)

print(f"Test Accuracy without augmentation: {acc_plain:.4f}")
print(f"Test Accuracy with augmentation: {acc_aug:.4f}")

# -------------------------------
# 6️⃣ Plot training curves
# -------------------------------
def plot_history(hist1, hist2):
    plt.figure(figsize=(12,5))

    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(hist1.history['accuracy'], label='Train Plain')
    plt.plot(hist1.history['val_accuracy'], label='Val Plain')
    plt.plot(hist2.history['accuracy'], label='Train Aug')
    plt.plot(hist2.history['val_accuracy'], label='Val Aug')
    plt.title("Accuracy Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1,2,2)
    plt.plot(hist1.history['loss'], label='Train Plain')
    plt.plot(hist1.history['val_loss'], label='Val Plain')
    plt.plot(hist2.history['loss'], label='Train Aug')
    plt.plot(hist2.history['val_loss'], label='Val Aug')
    plt.title("Loss Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()

plot_history(history_plain, history_aug)