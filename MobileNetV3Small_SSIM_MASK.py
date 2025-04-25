import os
import random
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, Concatenate, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight

from helper_func import get_video_prediction, evaluate_video_predictions, dual_input_generator, focal_loss
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

# GPU config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
print("Num GPUs Available:", len(physical_devices))

# Seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Data generators
batch_size = 16
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,
    width_shift_range=0.03,
    height_shift_range=0.03,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='reflect'
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('train', target_size=(224, 224), batch_size=batch_size, class_mode='binary')
val_generator = val_datagen.flow_from_directory('val', target_size=(224, 224), batch_size=batch_size, class_mode='binary', shuffle=False)
test_generator = test_datagen.flow_from_directory('test', target_size=(224, 224), batch_size=batch_size, class_mode='binary', shuffle=False)

# Class weights
class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_generator.classes)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print("Class weights:", class_weight_dict)

# Generator signature
def get_generator_signature():
    image_spec = tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32)
    ssim_spec = tf.TensorSpec(shape=(None, 224, 224, 1), dtype=tf.float32)
    ssim_stats_spec = tf.TensorSpec(shape=(None, 2), dtype=tf.float32)
    label_spec = tf.TensorSpec(shape=(None,), dtype=tf.float32)
    return ((image_spec, ssim_spec, ssim_stats_spec), label_spec)

train_generator_dual = tf.data.Dataset.from_generator(lambda: dual_input_generator(train_generator, 'train_ssim', 'train_ssim_var_mean'), output_signature=get_generator_signature())
val_generator_dual = tf.data.Dataset.from_generator(lambda: dual_input_generator(val_generator, 'val_ssim', 'val_ssim_var_mean'), output_signature=get_generator_signature())
test_generator_dual = tf.data.Dataset.from_generator(lambda: dual_input_generator(test_generator, 'test_ssim', 'test_ssim_var_mean'), output_signature=get_generator_signature())

# Model architecture
input_shape = (224, 224)
rgb_input = Input(shape=(*input_shape, 3), name="rgb_input")
base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_tensor=rgb_input)
base_model.trainable = True
x1 = GlobalAveragePooling2D()(base_model.output)
x1 = Dropout(0.2)(x1)  
x1 = Dense(1, activation='sigmoid')(x1)

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

checkpoint_cb = ModelCheckpoint("mobilenetv3small_ssim_model.h5", monitor="val_loss", save_best_only=True, mode="min", verbose=1)
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

# Test
threshold = 0.5
predictions = model.predict(test_generator_dual, steps=len(test_generator), verbose=1)
video_true_value, video_predictions_binary, video_predictions_probs = get_video_prediction(predictions, threshold, test_generator)
metrics = evaluate_video_predictions(y_true=video_true_value, y_pred_probs=video_predictions_probs, y_pred_binary=video_predictions_binary, class_names=["REAL", "FAKE"], model_name="Deepfake Detector MobileNetV3Small SSIM")
