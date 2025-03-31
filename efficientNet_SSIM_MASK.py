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
from helper_func import get_video_prediction, evaluate_video_predictions, dual_input_generator

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

import tensorflow as tf
import os
import numpy as np
import cv2


# Create final generators with output signatures
# def get_generator_signature():
#     # Define the output signature
#     image_spec = tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32)
#     ssim_spec = tf.TensorSpec(shape=(None, 224, 224, 1), dtype=tf.float32)
#     label_spec = tf.TensorSpec(shape=(None,), dtype=tf.float32)
    
#     return ((image_spec, ssim_spec), label_spec)

# # Create generators with proper output signatures
# train_generator_dual = tf.data.Dataset.from_generator(
#     lambda: dual_input_generator(train_generator, 'train_ssim'),
#     output_signature=get_generator_signature()
# )

# val_generator_dual = tf.data.Dataset.from_generator(
#     lambda: dual_input_generator(val_generator, 'val_ssim'),
#     output_signature=get_generator_signature()
# )

# test_generator_dual = tf.data.Dataset.from_generator(
#     lambda: dual_input_generator(test_generator, 'test_ssim'),
#     output_signature=get_generator_signature()
# )

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


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Flatten

input_shape = (224, 224)
rgb_input = Input(shape=(*input_shape, 3), name="rgb_input")  # Named for clarity

# Load EfficientNetB0 as the base model with pre-trained weights, excluding the top layer
base_model_rgb = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*input_shape, 3))

# Make EfficientNetB0 model trainable
base_model_rgb.trainable = True  # 

x1 = GlobalAveragePooling2D()(base_model_rgb.output)  # Global Average Pooling to reduce spatial dimensions
x1 = Dense(1, activation='sigmoid')(x1)

# 2. SSIM Branch
ssim_input = Input(shape=(*input_shape, 1), name="ssim_input")  # Named for clarity
x2 = Conv2D(16, (3,3), activation='relu')(ssim_input)
x2 = MaxPooling2D(4)(x2)
x2 = Conv2D(32, (3,3), activation='relu')(x2)
x2 = MaxPooling2D(4)(x2)
x2 = Conv2D(64, (3,3), activation='relu')(x2)
x2 = GlobalAveragePooling2D()(x2)  # Ensures correct shape

# 3. SSIM Statistics Branch (Mean & Variance)
ssim_stats_input = Input(shape=(2,), name="ssim_stats_input")  # Mean and variance as a 2D vector
x3 = Dense(16, activation='relu')(ssim_stats_input)  # Small FC network
x3 = Dense(8, activation='relu')(x3)

# 3. Merge both branches
combined = Concatenate()([x1, x2, x3])
# 4. Classification head
output = Dense(1, activation='sigmoid')(combined)

# 5. Build model
model = Model(inputs=[rgb_input, ssim_input, ssim_stats_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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
                                  patience=10,  # Stop if val_loss doesn't improve for 5 epochs
                                  restore_best_weights=True, 
                                  verbose=1)


history = model.fit(
    train_generator_dual,
    steps_per_epoch=len(train_generator),  # Use original generator's length
    validation_data=val_generator_dual,
    validation_steps=len(val_generator),   # Use original generator's length
    epochs=1,
    callbacks=[checkpoint_cb, early_stopping_cb],
    class_weight=class_weight_dict,
    verbose=1
)

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

mapping_label = {0: 'REAL', 1: 'FAKE'}

# ... [Previous code remains the same until predictions] ...

from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear



# Define the exit condition
while True:
    # Get a test batch from dual generator
    (test_images, test_ssim, test_ssim_stats), test_labels = next(test_generator_dual.as_numpy_iterator())

    # Select example
    sample_idx = 0
    rgb_image = test_images[sample_idx]
    ssim_map = test_ssim[sample_idx]
    true_label = int(test_labels[sample_idx])
    ssim_stats = test_ssim_stats[sample_idx]
    # Get prediction
    sample_prediction = model.predict([
        np.expand_dims(rgb_image, 0), 
        np.expand_dims(ssim_map, 0),
        np.expand_dims(ssim_stats, 0)  # Use combined stats tensor
    ])[0][0]
    predicted_class = 1 if sample_prediction > threshold else 0

    # Create GradCAM object with modifier for binary classification
    gradcam = Gradcam(model, model_modifier=ReplaceToLinear())

    def loss(output):
        """Simple loss for binary classification"""
        return output

    # Call Grad-CAM
    heatmap = gradcam(
        loss,
        [np.expand_dims(rgb_image, axis=0), np.expand_dims(ssim_map, axis=0), np.expand_dims(ssim_stats, axis=0)],  # Ensure correct input format
        penultimate_layer='top_conv',
        seek_penultimate_conv_layer=True,
        expand_cam=False,  # Disable zooming for debugging
        normalize_cam=True  # Optional: set to True to normalize heatmap
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
    plt.show()
    
    # Ask user if they are satisfied
    user_input = input("Do you want to see another plot? Type 'exit' to stop, anything else to continue: ").strip().lower()
    if user_input == 'exit':
        break