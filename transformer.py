import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import mixed_precision
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
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
        #mixed_precision.set_global_policy('mixed_float16')
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

from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_generator.classes)

# Convert to dictionary format for Keras
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

print("Computed class weights:", class_weight_dict)




threshold = 0.5


from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Dropout, Input
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
def vit_model(in_shape,num_classes):
    inputs = keras.Input(shape=in_shape)
    
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
    # Classify output
    outputs = layers.Dense(num_classes, activation='sigmoid')(features)
    
    model = keras.models.Model(inputs=inputs, outputs=[outputs])
    return model

vit_classifier = vit_model((224,224,3), 1)



import tensorflow.keras.backend as K

def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)  
        loss = -y_true * alpha * K.pow(1 - y_pred, gamma) * K.log(y_pred) - (1 - y_true) * (1 - alpha) * K.pow(y_pred, gamma) * K.log(1 - y_pred)
        return K.mean(loss)
    return loss

import tensorflow_addons as tfa
vit_classifier.compile(optimizer='adam', loss=tfa.losses.SigmoidFocalCrossEntropy(reduction="sum"), metrics=['accuracy'])
vit_classifier.summary()

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


history = vit_classifier.fit(
    train_generator,
    steps_per_epoch=len(train_generator),  # Use original generator's length
    validation_data=val_generator,
    validation_steps=len(val_generator),   # Use original generator's length
    epochs=200,
    callbacks=[checkpoint_cb, early_stopping_cb],
    class_weight=class_weight_dict,
    verbose=1
)

predictions = vit_classifier.predict(test_generator, steps=len(test_generator), verbose=1)

video_true_value, video_predictions_binary, video_predictions_probs = get_video_prediction(predictions, threshold, test_generator)


# Evaluate
metrics = evaluate_video_predictions(
    y_true=video_true_value,
    y_pred_probs = video_predictions_probs,
    y_pred_binary=video_predictions_binary,

    class_names=["REAL", "FAKE"],
    model_name="Deepfake Detector"
)
