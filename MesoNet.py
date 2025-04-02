import os
import random
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# Egendefinerte funksjoner
from helper_func import get_video_prediction, evaluate_video_predictions

# Bruk GPU hvis tilgjengelig
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
    rescale=1./255,              # Normalize pixel values to [0,1]
    
    # Mild geometric transformations (to avoid distorting faces)
    rotation_range=5,            # Reduce rotation to prevent unnatural face angles
    width_shift_range=0.03,      # Small shifts to avoid cropping face out
    height_shift_range=0.03,     
    # Controlled distortions
    zoom_range=0.05,             # Slight zoom without major distortion
    horizontal_flip=True,        # Keep flipping (deepfakes can be mirrored)

    fill_mode='reflect'          # Avoid unnatural padding artifacts
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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam

input_shape = (224, 224, 3)

# Build the model
model = Sequential([
    Conv2D(8, (3, 3), padding='same', activation='relu', input_shape=input_shape),
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
    Dense(1, activation='sigmoid')
])

# Compile the model
# model.compile(optimizer=Adam(learning_rate=0.001),
#               loss='mean_squared_error',
#               metrics=['accuracy'])
import tensorflow.keras.backend as K

def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)  
        loss = -y_true * alpha * K.pow(1 - y_pred, gamma) * K.log(y_pred) - (1 - y_true) * (1 - alpha) * K.pow(y_pred, gamma) * K.log(1 - y_pred)
        return K.mean(loss)
    return loss

model.compile(optimizer='adam', loss=focal_loss(alpha=0.25, gamma=2.0), metrics=['accuracy'])

# Print model summary
model.summary()

# Callbacks
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint_cb = ModelCheckpoint("meso_model.h5", monitor="val_loss", save_best_only=True, mode="min", verbose=1)
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
    y_pred_probs = video_predictions_probs,
    y_pred_binary=video_predictions_binary,

    class_names=["REAL", "FAKE"],
    model_name="Deepfake Detector"
)
