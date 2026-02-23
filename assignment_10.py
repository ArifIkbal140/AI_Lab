"""
Assignment 10

10. Write a report in pdf format using any Latex system after:
● training a binary classifier, based on the pre-trained VGG16, by transfer learning
  and fine tuning.
● showing the effect of fine-tuning:
    i.   whole pre-trained VGG16
    ii.  partial pre-trained VGG16

Dataset Path (Kaggle):
/kaggle/input/dogs-vs-cats/
"""

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# ============================================================
# 1. DATASET PREPARATION (Kaggle Path)
# ============================================================

DATASET_PATH = "/kaggle/input/dogs-vs-cats"

# Kaggle dataset usually has train folder only.
# We split using validation_split

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# ============================================================
# MODEL CREATION FUNCTION
# ============================================================

def create_model(mode="freeze_all"):

    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224,224,3)
    )

    # -------- Transfer Learning --------
    if mode == "freeze_all":
        base_model.trainable = False

    # -------- Partial Fine-Tuning --------
    elif mode == "partial":
        base_model.trainable = True
        for layer in base_model.layers[:-4]:
            layer.trainable = False

    # -------- Full Fine-Tuning --------
    elif mode == "full":
        base_model.trainable = True

    x = base_model.output
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=base_model.input, outputs=output)

    if mode == "freeze_all":
        lr = 1e-3
    else:
        lr = 1e-5

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# ============================================================
# 2. TRANSFER LEARNING (All layers frozen)
# ============================================================

print("\nTraining: Transfer Learning (All VGG16 layers frozen)")
model_transfer = create_model("freeze_all")

history_transfer = model_transfer.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator
)

# ============================================================
# 3. PARTIAL FINE-TUNING
# ============================================================

print("\nTraining: Partial Fine-Tuning (Last 4 layers unfrozen)")
model_partial = create_model("partial")

history_partial = model_partial.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator
)

# ============================================================
# 4. FULL FINE-TUNING
# ============================================================

print("\nTraining: Full Fine-Tuning (All VGG16 layers unfrozen)")
model_full = create_model("full")

history_full = model_full.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator
)

# ============================================================
# 5. SHOW EFFECT OF FINE-TUNING
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(history_transfer.history['val_accuracy'], label='Transfer Learning')
plt.plot(history_partial.history['val_accuracy'], label='Partial Fine-Tuning')
plt.plot(history_full.history['val_accuracy'], label='Full Fine-Tuning')

plt.title("Effect of Fine-Tuning on Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.show()