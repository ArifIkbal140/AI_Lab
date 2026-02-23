
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
from tensorflow.keras.applications.resnet50 import preprocess_input as res_pre
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mob_pre

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model



IMG_PATH = "/content/drive/MyDrive/data/test/Khata/khata_image_100.jpg "  

def load_img(path, size=(224,224)):
    img = image.load_img(path, target_size=size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img


# -----------------------------
# Function to Display Feature Maps
# -----------------------------
def visualize_feature_maps(model, img_array, preprocess_func, layer_names, model_name):

    img = preprocess_func(img_array.copy())

  
    outputs = [model.get_layer(name).output for name in layer_names]
    feature_model = Model(inputs=model.input, outputs=outputs)

    feature_maps = feature_model.predict(img)

    for fmap, lname in zip(feature_maps, layer_names):

        n_features = fmap.shape[-1]
        size = fmap.shape[1]

        display_maps = min(8, n_features)

        plt.figure(figsize=(15,3))
        plt.suptitle(f"{model_name} - Layer: {lname}", fontsize=14)

        for i in range(display_maps):
            ax = plt.subplot(1, display_maps, i+1)

            x = fmap[0, :, :, i]
            x -= x.mean()
            x /= (x.std() + 1e-5)
            x *= 64
            x += 128
            x = np.clip(x, 0, 255).astype('uint8')

            plt.imshow(x, cmap='viridis')
            plt.axis("off")

        plt.show()




vgg_model = VGG16(weights='imagenet', include_top=False)

vgg_layers = [
    "block1_conv1",   # early
    "block3_conv1",   # middle
    "block5_conv1"    # deep
]

img_vgg = load_img(IMG_PATH)

visualize_feature_maps(
    model=vgg_model,
    img_array=img_vgg,
    preprocess_func=vgg_pre,
    layer_names=vgg_layers,
    model_name="VGG16"
)


# ==========================================
# 2️⃣ ResNet50
# ==========================================

resnet_model = ResNet50(weights='imagenet', include_top=False)

resnet_layers = [
    "conv1_conv",        # early
    "conv3_block1_out",  # middle
    "conv5_block1_out"   # deep
]

img_res = load_img(IMG_PATH)

visualize_feature_maps(
    model=resnet_model,
    img_array=img_res,
    preprocess_func=res_pre,
    layer_names=resnet_layers,
    model_name="ResNet50"
)


# ==========================================
# 3️⃣ MobileNetV2
# ==========================================

mobilenet_model = MobileNetV2(weights='imagenet', include_top=False)

mobilenet_layers = [
    "Conv1",                # early
    "block_6_expand_relu",  # middle
    "block_13_expand_relu"  # deep
]

img_mob = load_img(IMG_PATH)

visualize_feature_maps(
    model=mobilenet_model,
    img_array=img_mob,
    preprocess_func=mob_pre,
    layer_names=mobilenet_layers,
    model_name="MobileNetV2"
)