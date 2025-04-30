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

from helper_func import get_video_prediction, evaluate_video_predictions, focal_loss

gpu = True
if gpu:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print(os.environ['CUDA_VISIBLE_DEVICES'])  

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

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

from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_generator.classes)

class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

print("Computed class weights:", class_weight_dict)

threshold = 0.5

from vit_keras import vit

vit_classifier = vit.vit_b16(
    image_size=224,
    activation='sigmoid',
    pretrained=True,
    include_top=True,
    pretrained_top=False,
    classes=1
)
vit_classifier.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])
vit_classifier.summary()

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint_cb = ModelCheckpoint("best_model.h5", 
                                monitor="val_loss", 
                                save_best_only=True, 
                                mode="min", 
                                verbose=1)

early_stopping_cb = EarlyStopping(monitor="val_loss", 
                                  patience=50,  
                                  restore_best_weights=True, 
                                  verbose=1)


history = vit_classifier.fit(
    train_generator,
    steps_per_epoch=len(train_generator),  
    validation_data=val_generator,
    validation_steps=len(val_generator),  
    epochs=200,
    callbacks=[checkpoint_cb, early_stopping_cb],
    class_weight=class_weight_dict,
    verbose=1
)

predictions = vit_classifier.predict(test_generator, steps=len(test_generator), verbose=1)

video_true_value, video_predictions_binary, video_predictions_probs = get_video_prediction(predictions, threshold, test_generator)


metrics = evaluate_video_predictions(
    y_true=video_true_value,
    y_pred_probs = video_predictions_probs,
    y_pred_binary=video_predictions_binary,

    class_names=["REAL", "FAKE"],
    model_name="Deepfake Detector (Transformer)"
)
mapping_label = {0: 'REAL', 1: 'FAKE'}
from sklearn.manifold import TSNE

feature_model = Model(inputs=vit_classifier.input, outputs=vit_classifier.layers[-2].output)

all_images = []
all_labels = []

for i in range(len(test_generator)):
    x_batch, y_batch = test_generator[i]
    all_images.append(x_batch)
    all_labels.append(y_batch)

all_images = np.concatenate(all_images, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

features = feature_model.predict(all_images, batch_size=batch_size, verbose=1)

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
count = 0
