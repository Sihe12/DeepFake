import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import pandas as pd
import random

# load functions from helper_func.py
from helper_func import get_video_prediction, evaluate_video_predictions, dual_input_generator, focal_loss

gpu = True
# Use gpu if available
if gpu:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print(os.environ['CUDA_VISIBLE_DEVICES'])  # Check the value

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

batch_size = 16
# Create ImageDataGenerators
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
# Create generators
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

import tensorflow as tf
import os
import numpy as np
import cv2

def get_generator_signature():
    # Define output signature
    image_spec = tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32)
    ssim_spec = tf.TensorSpec(shape=(None, 224, 224, 1), dtype=tf.float32)
    ssim_stats_spec = tf.TensorSpec(shape=(None, 2), dtype=tf.float32)  # Changed from two separate specs to one (batch_size, 2)
    label_spec = tf.TensorSpec(shape=(None,), dtype=tf.float32)

    return ((image_spec, ssim_spec, ssim_stats_spec), label_spec)

# Create generators with proper output signatures
train_generator_dual = tf.data.Dataset.from_generator(
    lambda: dual_input_generator(train_generator, 'train_ssim', 'train_ssim_var_mean'),
    output_signature=get_generator_signature()
)

val_generator_dual = tf.data.Dataset.from_generator(
    lambda: dual_input_generator(val_generator, 'val_ssim', 'val_ssim_var_mean'),
    output_signature=get_generator_signature()
)

test_generator_dual = tf.data.Dataset.from_generator(
    lambda: dual_input_generator(test_generator, 'test_ssim', 'test_ssim_var_mean'),
    output_signature=get_generator_signature()
)

from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_generator.classes)

# Convert to dictionary format for Keras
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

print("Computed class weights:", class_weight_dict)




threshold = 0.5


from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, GlobalAveragePooling2D, Concatenate, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model, Sequential
import keras
from keras import layers

from vit_keras import vit

rgb_input = Input(shape=(224, 224, 3), name="rgb_input")  # <-- This is what you need
x1 = vit.vit_b16(
    image_size=224,
    activation='sigmoid',
    pretrained=True,
    include_top=False,  # Changed to False since we're adding our own head
    pretrained_top=False,
    classes=1
)(rgb_input)

input_shape = (224, 224)
# 2. SSIM Branch
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

x2 = ssim_branch(ssim_input)  # Correctly apply the input tensor


# 3. SSIM Statistics Branch (Mean & Variance)
ssim_stats_input = Input(shape=(2,), name="ssim_stats_input")
ssim_stats_branch = Sequential([
    Dense(16, activation='relu', input_shape=(2,)),
    Dense(8, activation='relu')
])

x3 = ssim_stats_branch(ssim_stats_input)  # Correctly apply the input tensor

# 4. Merge All Branches
combined = Concatenate()([x1, x2, x3])
# 4. Classification head
output = Dense(1, activation='sigmoid')(combined)

# 5. Build model
model = Model(inputs=[rgb_input, ssim_input, ssim_stats_input], outputs=output)
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])
# Print model summary
model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Define callbacks
checkpoint_cb = ModelCheckpoint("best_model.h5", 
                                monitor="val_loss", 
                                save_best_only=True, 
                                mode="min", 
                                verbose=1)

early_stopping_cb = EarlyStopping(monitor="val_loss", 
                                  patience=50,  # Stop if val_loss doesn't improve for 5 epochs
                                  restore_best_weights=True, 
                                  verbose=1)


history = model.fit(
    train_generator_dual,
    steps_per_epoch=len(train_generator),  # Use original generator's length
    validation_data=val_generator_dual,
    validation_steps=len(val_generator),   # Use original generator's length
    epochs=200,
    callbacks=[checkpoint_cb, early_stopping_cb],
    class_weight=class_weight_dict,
    verbose=1
)

predictions = model.predict(test_generator_dual, steps=len(test_generator), verbose=1)

video_true_value, video_predictions_binary, video_predictions_probs = get_video_prediction(predictions, threshold, test_generator)


# Evaluate
metrics = evaluate_video_predictions(
    y_true=video_true_value,
    y_pred_probs = video_predictions_probs,
    y_pred_binary=video_predictions_binary,

    class_names=["REAL", "FAKE"],
    model_name="Deepfake Detector (Transformer SSIM)"
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
plt.show()
plt.close()