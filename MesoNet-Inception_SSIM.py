import os
import random
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
                                     LeakyReLU, BatchNormalization, GlobalAveragePooling2D, Concatenate)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from helper_func import get_video_prediction, evaluate_video_predictions, dual_input_generator
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

# GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
print("Num GPUs Available:", len(physical_devices))

# Seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Data Generators
batch_size = 16
input_shape = (224, 224)

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=5,
                                   width_shift_range=0.03,
                                   height_shift_range=0.03,
                                   zoom_range=0.05,
                                   horizontal_flip=True,
                                   fill_mode='reflect')

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('train', target_size=input_shape, batch_size=batch_size, class_mode='binary')
val_generator = val_test_datagen.flow_from_directory('val', target_size=input_shape, batch_size=batch_size, class_mode='binary', shuffle=False)
test_generator = val_test_datagen.flow_from_directory('test', target_size=input_shape, batch_size=batch_size, class_mode='binary', shuffle=False)

class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_generator.classes)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print("Class weights:", class_weight_dict)

# SSIM-aware generator

def get_generator_signature():
    image_spec = tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32)
    ssim_spec = tf.TensorSpec(shape=(None, 224, 224, 1), dtype=tf.float32)
    ssim_stats_spec = tf.TensorSpec(shape=(None, 2), dtype=tf.float32)
    label_spec = tf.TensorSpec(shape=(None,), dtype=tf.float32)
    return ((image_spec, ssim_spec, ssim_stats_spec), label_spec)

def meso_inception_block(x, filters):
    # Reduksjon før dilated convs
    path1 = Conv2D(filters, (1, 1), padding='same')(x)
    path1 = BatchNormalization()(path1)
    path1 = LeakyReLU(alpha=0.1)(path1)
    path1 = Conv2D(filters, (3, 3), dilation_rate=1, padding='same')(path1)
    path1 = BatchNormalization()(path1)
    path1 = LeakyReLU(alpha=0.1)(path1)

    path2 = Conv2D(filters, (1, 1), padding='same')(x)
    path2 = BatchNormalization()(path2)
    path2 = LeakyReLU(alpha=0.1)(path2)
    path2 = Conv2D(filters, (3, 3), dilation_rate=2, padding='same')(path2)
    path2 = BatchNormalization()(path2)
    path2 = LeakyReLU(alpha=0.1)(path2)

    # Skip connection
    skip = Conv2D(filters, (1, 1), padding='same')(x)
    skip = BatchNormalization()(skip)
    skip = LeakyReLU(alpha=0.1)(skip)

    # Samle alle paths
    x = Concatenate()([path1, path2, skip])
    return x

train_generator_dual = tf.data.Dataset.from_generator(
    lambda: dual_input_generator(train_generator, 'train_ssim', 'train_ssim_var_mean'),
    output_signature=get_generator_signature())
val_generator_dual = tf.data.Dataset.from_generator(
    lambda: dual_input_generator(val_generator, 'val_ssim', 'val_ssim_var_mean'),
    output_signature=get_generator_signature())
test_generator_dual = tf.data.Dataset.from_generator(
    lambda: dual_input_generator(test_generator, 'test_ssim', 'test_ssim_var_mean'),
    output_signature=get_generator_signature())

# Model Inputs
rgb_input = Input(shape=(224, 224, 3), name="rgb_input")
ssim_input = Input(shape=(224, 224, 1), name="ssim_input")
ssim_stats_input = Input(shape=(2,), name="ssim_stats_input")

# RGB Branch (MesoInception-style)
x = meso_inception_block(rgb_input, filters=8)
x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

x = meso_inception_block(x, filters=8)
x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

x = Conv2D(16, (5, 5), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

x = Conv2D(16, (5, 5), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(4, 4), padding='same')(x)

x1 = Flatten()(x)


# SSIM Branch (Functional)
ssim_x = Conv2D(16, (3, 3), activation='relu', padding='same')(ssim_input)
ssim_x = BatchNormalization()(ssim_x)
ssim_x = MaxPooling2D((2, 2))(ssim_x)

ssim_x = Conv2D(32, (3, 3), activation='relu', padding='same')(ssim_x)
ssim_x = BatchNormalization()(ssim_x)
ssim_x = MaxPooling2D((2, 2))(ssim_x)

ssim_x = Conv2D(64, (3, 3), activation='relu', padding='same')(ssim_x)
ssim_x = BatchNormalization()(ssim_x)
ssim_x = MaxPooling2D((2, 2))(ssim_x)

ssim_x = GlobalAveragePooling2D()(ssim_x)
x2 = Dense(64, activation='relu')(ssim_x)
x2 = Dropout(0.5)(x2)


# SSIM Stats Branch
ssim_stats_branch = Sequential([
    Dense(16, activation='relu', input_shape=(2,)),
    Dense(8, activation='relu')
])
x3 = ssim_stats_branch(ssim_stats_input)

# Combine and Classify
combined = Concatenate()([x1, x2, x3])
x = Dropout(0.5)(combined)
x = Dense(16)(x)
x = LeakyReLU(alpha=0.1)(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[rgb_input, ssim_input, ssim_stats_input], outputs=output)

# Loss
import tensorflow.keras.backend as K
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        loss = -y_true * alpha * K.pow(1 - y_pred, gamma) * K.log(y_pred) - \
               (1 - y_true) * (1 - alpha) * K.pow(y_pred, gamma) * K.log(1 - y_pred)
        return K.mean(loss)
    return loss

import tensorflow_addons as tfa
model.compile(optimizer='adam', loss=tfa.losses.SigmoidFocalCrossEntropy(reduction="sum"), metrics=['accuracy'])
model.summary()

# Callbacks
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint_cb = ModelCheckpoint("best_model_ssim_meso.h5", monitor="val_loss", save_best_only=True, mode="min", verbose=1)
early_stopping_cb = EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True, verbose=1)

# Train
history = model.fit(
    train_generator_dual,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator_dual,
    validation_steps=len(val_generator),
    epochs=200,
    callbacks=[checkpoint_cb, early_stopping_cb],
    class_weight=class_weight_dict,
    verbose=1
)

# Grad-CAM Example Visualization
threshold = 0.5
predictions = model.predict(test_generator_dual, steps=len(test_generator), verbose=1)

video_true_value, video_predictions_binary, video_predictions_probs = get_video_prediction(predictions, threshold, test_generator)


# Evaluate
metrics = evaluate_video_predictions(
    y_true=video_true_value,
    y_pred_probs = video_predictions_probs,
    y_pred_binary=video_predictions_binary,

    class_names=["REAL", "FAKE"],
    model_name="Deepfake Detector"
)


mapping_label = {0: 'REAL', 1: 'FAKE'}

from sklearn.manifold import TSNE

feature_model = Model(inputs=model.input, outputs=model.layers[-3].output)

all_images = []
all_labels = []

for (x_batch, y_batch) in test_generator_dual.take(len(test_generator)):
    rgb_batch, ssim_batch, stats_batch = x_batch
    all_images.append((rgb_batch, ssim_batch, stats_batch))
    all_labels.append(y_batch)

rgb_all = np.concatenate([x[0] for x in all_images], axis=0)
ssim_all = np.concatenate([x[1] for x in all_images], axis=0)
stats_all = np.concatenate([x[2] for x in all_images], axis=0)
all_labels = np.concatenate(all_labels, axis=0)

features = feature_model.predict([rgb_all, ssim_all, stats_all], batch_size=batch_size, verbose=1)

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=SEED)
features_2d = tsne.fit_transform(features)

plt.figure(figsize=(10, 7))
for label in np.unique(all_labels):
    idx = all_labels == label
    plt.scatter(features_2d[idx, 0], features_2d[idx, 1], label=mapping_label[int(label)], alpha=0.6)

plt.legend()
plt.title("TSNE of features")
plt.xlabel("TSNE component 1")
plt.ylabel("TSNE component 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("tsne_visualisering.png")
plt.close()
count = 0

while True:
    test_batch = next(test_generator_dual.as_numpy_iterator())
    (test_images, ssim_maps, ssim_stats), test_labels = test_batch

    sample_idx = 0
    rgb_image = test_images[sample_idx]
    ssim_map = ssim_maps[sample_idx]
    stats = ssim_stats[sample_idx]
    true_label = int(test_labels[sample_idx])

    sample_prediction = model.predict([
        np.expand_dims(rgb_image, 0),
        np.expand_dims(ssim_map, 0),
        np.expand_dims(stats, 0)
    ])[0][0]
    predicted_class = 1 if sample_prediction > threshold else 0

    penultimate_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            penultimate_layer_name = layer.name
            break

    gradcam = Gradcam(model, model_modifier=ReplaceToLinear())

    def loss_fn(output): 
        return output

    heatmap = gradcam(
        loss_fn,
        [np.expand_dims(rgb_image, axis=0),
        np.expand_dims(ssim_map, axis=0),
        np.expand_dims(stats, axis=0)],
        penultimate_layer=penultimate_layer_name,
        seek_penultimate_conv_layer=True,
        expand_cam=False,  # Viktig: slå av automatisk zooming
        normalize_cam=True  # Normaliser output
    )

    # Process heatmap
    heatmap = np.squeeze(heatmap)
    if len(heatmap.shape) > 2:  # If still multi-channel
        heatmap = heatmap[0]  # Take first channel (or np.mean(heatmap, axis=0))
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Simple overlay function if you don't have overlay_heatmap
    def simple_overlay(image, heatmap, alpha=0.5):
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return cv2.addWeighted(image, alpha, heatmap, 1-alpha, 0)

    # Visualization (same as before)
    rgb_image_uint8 = (rgb_image * 255).astype(np.uint8)
    overlay = simple_overlay(rgb_image_uint8, heatmap)

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_image_uint8)
    plt.title(f"Original\nTrue: {mapping_label[true_label]}\nPred: {mapping_label[predicted_class]} ({sample_prediction:.2f})")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"gradcam_overlay_{count}.png")
    #plt.show()
    count += 1
    
    # Ask user if they are satisfied
    user_input = input("Do you want to see another plot? Type 'exit' to stop, anything else to continue: ").strip().lower()
    if user_input == 'exit':
        break