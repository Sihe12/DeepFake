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
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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

image_size = 224 # We'll resize input images to this size
patch_size = 6 # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
embed_dim = 64
num_heads = 8
transformer_layers = 3 # Size of the transformer layers block
transfomer_units = [
    embed_dim* 2,
    embed_dim
]
mlp_head_units = [
    2048,
    1024
]

# implement multilayer percetron
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units,activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# create model
def vit_model(inputs,num_classes):
    
    # patching the input image into patches
    patching = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, padding='valid',name='patching')(inputs)
    patches = layers.Reshape((num_patches, embed_dim), name='pacthes')(patching)
    
    # Learnable positional embeddings
    position_embeddings = layers.Embedding(input_dim=num_patches, output_dim=embed_dim)(layers.Lambda(lambda x: tf.expand_dims(tf.range(num_patches), axis=0))(patches))

    # Add positional embeddings to patches
    patches = layers.Add()([patches,position_embeddings])
    
    # Create multiple layers of transformer block
    for _ in range(transformer_layers):
        # Create mulithead attention layer
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)(patches,patches)
        # skip connection 1
        x2 = layers.Add()([attention_output, patches])
        # Normalization 1
        x2 = layers.LayerNormalization(epsilon=1e-6)(x2)
        
        # Feed forward network
        x3 = mlp(x2, hidden_units=transfomer_units,dropout_rate=0.1)
        #Skip connection 2
        patches = layers.Add()([x3, x2])
        # Normalization 2
        patches = layers.LayerNormalization(epsilon=1e-6)(patches)
        
    representation = layers.Flatten()(patches)
    representation = layers.Dropout(0.5)(representation)
    
    features = mlp(representation,hidden_units=mlp_head_units,dropout_rate=0.5)
    # # Classify output
    # outputs = layers.Dense(num_classes, activation='sigmoid')(features)
    
    # model = keras.models.Model(inputs=inputs, outputs=[outputs])
    return features

input_shape = (224, 224)
rgb_input = Input(shape=(*input_shape, 3), name="rgb_input")

x1 = vit_model(rgb_input, 1)

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
import tensorflow.keras.backend as K

def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)  
        loss = -y_true * alpha * K.pow(1 - y_pred, gamma) * K.log(y_pred) - (1 - y_true) * (1 - alpha) * K.pow(y_pred, gamma) * K.log(1 - y_pred)
        return K.mean(loss)
    return loss

import tensorflow_addons as tfa
model.compile(optimizer='adam', loss=tfa.losses.SigmoidFocalCrossEntropy(reduction="sum"), metrics=['accuracy'])
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
    model_name="Deepfake Detector"
)

mapping_label = {0: 'REAL', 1: 'FAKE'}