# ==========================================
# Fully Connected Feedforward Neural Network (FCFNN)
# Training and Evaluation on MNIST
# ==========================================

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Preprocess dataset
# ------------------------------
def preprocess_data(trainX, trainY, testX, testY):
    print(f"trainX shape: {trainX.shape}, trainY shape: {trainY.shape}")
    print(f"testX shape: {testX.shape}, testY shape: {testY.shape}")

    trainX = trainX.astype('float32') / 255.0
    testX = testX.astype('float32') / 255.0

    trainY = to_categorical(trainY, 10)
    testY = to_categorical(testY, 10)

    print("After normalization and one-hot encoding:")
    print(f"trainX shape: {trainX.shape}, trainY shape: {trainY.shape}")
    print(f"testX shape: {testX.shape}, testY shape: {testY.shape}")
    return (trainX, trainY), (testX, testY)

# ------------------------------
# Build FCFNN model
# ------------------------------
def build_model(input_shape):
    inputs = Input(input_shape)
    x = Flatten()(inputs)
    h1 = Dense(128, activation='relu')(x)
    h2 = Dense(64, activation='relu')(h1)
    h3 = Dense(128, activation='relu')(h2)
    outputs = Dense(10, activation='softmax')(h3)

    model = Model(inputs, outputs)
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ------------------------------
# Train the model
# ------------------------------
def train_model(model, trainX, trainY, epochs=15):
    history = model.fit(
        trainX, trainY,
        validation_split=0.2,
        batch_size=64,
        epochs=epochs
    )
    return history

# ------------------------------
# Plot training loss
# ------------------------------
def loss_curve(history):
    plt.figure(figsize=(8,6))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# ------------------------------
# Sample predictions visualization
# ------------------------------
def sample_prediction_plot(testX, testY_labels, predY_labels):
    plt.figure(figsize=(10,10))
    num_samples, rows, cols = 25, 5, 5
    for i in range(min(num_samples, len(testX))):
        plt.subplot(rows, cols, i+1)
        plt.axis('off')
        plt.imshow(testX[i], cmap='gray')
        color = 'green' if testY_labels[i] == predY_labels[i] else 'red'
        plt.title(f'True: {testY_labels[i]}\nPred: {predY_labels[i]}', color=color)
    plt.suptitle("Sample Predictions from MNIST Test Set")
    plt.tight_layout()
    plt.show()

# ------------------------------
# Main function
# ------------------------------
def main():
    # Load MNIST dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()

    # Preprocess
    (trainX, trainY), (testX, testY) = preprocess_data(trainX, trainY, testX, testY)

    # Build and train model
    input_shape = trainX.shape[1:]
    model = build_model(input_shape)
    history = train_model(model, trainX, trainY, epochs=15)

    # Plot loss curve
    loss_curve(history)

    # Evaluate on MNIST test set
    predY = model.predict(testX)
    predY_labels = np.argmax(predY, axis=1)
    testY_labels = np.argmax(testY, axis=1)
    test_accuracy = accuracy_score(testY_labels, predY_labels)
    print(f"Accuracy on MNIST Test Set: {test_accuracy:.4f}")

    sample_prediction_plot(testX, testY_labels, predY_labels)

# ------------------------------
# Run main
# ------------------------------
if __name__ == "__main__":
    main()