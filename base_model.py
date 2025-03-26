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
from helper_func import get_video_prediction, evaluate_video_predictions

gpu = True
# Use gpu if available
if gpu:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
    rescale=1./255,              # Normalize pixel values to [0, 1]
    rotation_range=10,           # Randomly rotate images by up to 10 degrees
    width_shift_range=0.1,       # Randomly shift images horizontally by 10% of the width
    height_shift_range=0.1,      # Randomly shift images vertically by 10% of the height
    shear_range=0.2,             # Apply shearing transformations
    zoom_range=0.1,              # Randomly zoom in or out by 20%
    horizontal_flip=True,        # Randomly flip images horizontally
    fill_mode='nearest'          # Fill missing pixels after transformations,

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
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    'test', 
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary'
)

from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_generator.classes)

# Convert to dictionary format for Keras
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

print("Computed class weights:", class_weight_dict)




threshold = 0.5


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

input_shape = (224, 224, 3)

model = Sequential([
    # First Conv Layer
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
    MaxPooling2D((2, 2)),

    # Second Conv Layer
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    # Third Conv Layer
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    # Flatten and Fully Connected Layers
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Regularization to prevent overfitting
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Define callbacks
checkpoint_cb = ModelCheckpoint("best_model.h5", 
                                monitor="val_loss", 
                                save_best_only=True, 
                                mode="min", 
                                verbose=1)

early_stopping_cb = EarlyStopping(monitor="val_loss", 
                                  patience=10,  # Stop if val_loss doesn't improve for 5 epochs
                                  restore_best_weights=True, 
                                  verbose=1)


history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),  # Use original generator's length
    validation_data=val_generator,
    validation_steps=len(val_generator),   # Use original generator's length
    epochs=1,
    callbacks=[checkpoint_cb, early_stopping_cb],
    class_weight=class_weight_dict,
    verbose=1
)

predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)

predicted_classes = (predictions > threshold).astype(int).flatten()  # Convert to binary (0/1)


video_predictions, video_true_value = get_video_prediction(predicted_classes, threshold, test_generator)


# Evaluate
metrics = evaluate_video_predictions(
    y_true=video_predictions,
    y_pred=video_true_value,
    class_names=["REAL", "FAKE"],
    model_name="Deepfake Detector"
)

mapping_label = {0: 'REAL', 1: 'FAKE'}

# ... [Previous code remains the same until predictions] ...

# Get a batch of test images
test_images, test_labels = next(test_generator)

# Select the first image from the batch for Grad-CAM visualization
sample_idx = 0  # You can change this to view different samples
rgb_image = test_images[sample_idx]
true_label = int(test_labels[sample_idx])  # Convert to 0 or 1

# Get model's prediction for this image
sample_prediction = model.predict(np.expand_dims(rgb_image, axis=0))[0][0]
predicted_class = 1 if sample_prediction > threshold else 0

from tf_keras_vis.gradcam import Gradcam
# Create GradCAM object
gradcam = Gradcam(model)

def loss(output):
    """Simple loss for binary classification"""
    return output

# Generate heatmap - use last conv layer ('conv2d_2' in your model)
heatmap = gradcam(loss,
                 np.expand_dims(rgb_image, axis=0),
                 penultimate_layer='conv2d_2')  # Use your last conv layer name

# Process heatmap
heatmap = np.squeeze(heatmap)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

# Create visualization
rgb_image_uint8 = (rgb_image * 255).astype(np.uint8)  # Convert to 0-255 range

# Simple overlay function if you don't have overlay_heatmap
def simple_overlay(image, heatmap, alpha=0.5):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, alpha, heatmap, 1-alpha, 0)

overlay = simple_overlay(rgb_image_uint8, heatmap)

# Plot results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(rgb_image_uint8)
plt.title(f"Original\nTrue: {mapping_label[true_label]}\nPred: {mapping_label[predicted_class]} ({sample_prediction:.2f})")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(overlay)
plt.title("Overlay")
plt.axis('off')

plt.tight_layout()
plt.show()