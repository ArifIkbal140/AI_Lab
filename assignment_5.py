import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10


# ==========================================
# 1️⃣ Build CNN Model
# ==========================================

def build_cnn(input_shape, num_classes):

    inputs = Input(shape=input_shape, name='input_layer')

    # Convolution Block 1
    x = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2,2))(x)

    # Convolution Block 2
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)

    # Convolution Block 3
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)

    # Fully Connected
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name="CNN_Classifier")

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ==========================================
# 2️⃣ Train & Evaluate Function
# ==========================================

def train_and_evaluate(dataset_name, x_train, y_train,
                       x_test, y_test, input_shape):

    print(f"\nTraining on {dataset_name}")

    # Normalize
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    model = build_cnn(input_shape, 10)

    model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=128,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    print(f"{dataset_name} Test Accuracy:", test_acc)


# ==========================================
# 3️⃣ MNIST Dataset
# ==========================================

(x_train_m, y_train_m), (x_test_m, y_test_m) = mnist.load_data()

# Add channel dimension
x_train_m = np.expand_dims(x_train_m, -1)
x_test_m = np.expand_dims(x_test_m, -1)

train_and_evaluate(
    "MNIST",
    x_train_m, y_train_m,
    x_test_m, y_test_m,
    input_shape=(28, 28, 1)
)


# ==========================================
# 4️⃣ Fashion MNIST Dataset
# ==========================================

(x_train_f, y_train_f), (x_test_f, y_test_f) = fashion_mnist.load_data()

x_train_f = np.expand_dims(x_train_f, -1)
x_test_f = np.expand_dims(x_test_f, -1)

train_and_evaluate(
    "Fashion MNIST",
    x_train_f, y_train_f,
    x_test_f, y_test_f,
    input_shape=(28, 28, 1)
)


# ==========================================
# 5️⃣ CIFAR-10 Dataset
# ==========================================

(x_train_c, y_train_c), (x_test_c, y_test_c) = cifar10.load_data()

# Flatten labels
y_train_c = y_train_c.flatten()
y_test_c = y_test_c.flatten()

train_and_evaluate(
    "CIFAR-10",
    x_train_c, y_train_c,
    x_test_c, y_test_c,
    input_shape=(32, 32, 3)
)