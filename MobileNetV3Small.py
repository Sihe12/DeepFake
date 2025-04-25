import os
import random
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

# Egendefinerte funksjoner
from helper_func import get_video_prediction, evaluate_video_predictions

# Bruk GPU hvis tilgjengelig
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
print("Num GPUs Available:", len(physical_devices))

# Sett seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Datageneratorer
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

train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary'
)
val_generator = val_datagen.flow_from_directory(
    'val',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)
test_generator = test_datagen.flow_from_directory(
    'test',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Beregn class weights
class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_generator.classes)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print("Class weights:", class_weight_dict)

# Bygg modellen
input_shape = (224, 224, 3)
input_tensor = Input(shape=input_shape)

base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_tensor=input_tensor)
base_model.trainable = True

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)  
x = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=x)

# Focal loss
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

# Print model summary
model.summary()

# Callbacks
checkpoint_cb = ModelCheckpoint("mobilenetv3small_model.h5", monitor="val_loss", save_best_only=True, mode="min", verbose=1)
early_stopping_cb = EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True, verbose=1)

# Tren modellen
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    epochs=200,
    callbacks=[checkpoint_cb, early_stopping_cb],
    class_weight=class_weight_dict,
    verbose=1
)

# Evaluer på testsettet
threshold = 0.5
predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)

video_true_value, video_predictions_binary, video_predictions_probs = get_video_prediction(predictions, threshold, test_generator)

# Evaluate
metrics = evaluate_video_predictions(
    y_true=video_true_value,
    y_pred_probs=video_predictions_probs,
    y_pred_binary=video_predictions_binary,
    class_names=["REAL", "FAKE"],
    model_name="Deepfake Detector MobileNetV3Small"
)

