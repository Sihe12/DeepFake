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
from helper_func import get_video_prediction, evaluate_video_predictions, dual_input_generator, focal_loss
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tensorflow.keras.regularizers import l2
from sklearn.manifold import TSNE

# GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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

# RGB Branch (MesoNet-style)
rgb_branch = Sequential([
    Conv2D(8, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    
    Conv2D(8, (5, 5), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    
    Conv2D(16, (5, 5), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    
    Conv2D(16, (5, 5), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(4, 4), padding='same'),
    
    Flatten(),
    Dropout(0.5),
    Dense(16),
    LeakyReLU(alpha=0.1),
    Dropout(0.5),
])

x1 = rgb_branch(rgb_input)
ssim_input = Input(shape=(*input_shape, 1), name="ssim_input")
ssim_branch = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    GlobalAveragePooling2D(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
])
x2 = ssim_branch(ssim_input)

ssim_stats_input = Input(shape=(2,), name="ssim_stats_input")
ssim_stats_branch = Sequential([
    Dense(16, activation='relu', input_shape=(2,)),
    Dense(8, activation='relu')
])
x3 = ssim_stats_branch(ssim_stats_input)

combined = Concatenate()([x1, x2, x3])
output = Dense(1, activation='sigmoid')(combined)
model = Model(inputs=[rgb_input, ssim_input, ssim_stats_input], outputs=output)

model.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy'])
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

feature_model = Model(inputs=model.input, outputs=model.get_layer('concatenate').output)

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