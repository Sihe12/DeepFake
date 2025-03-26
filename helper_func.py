import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

def get_video_prediction(predictions, threshold=0.5, test_generator=None):
    # Extract video names from image filenames
    image_filenames = test_generator.filenames  # From your test generator
    video_names = [os.path.splitext(os.path.basename(f))[0].split('_')[0] for f in image_filenames]
    

    true_value = test_generator.classes  # True labels
    # Use the equal video names index to get the predictions for each video
    video_predictions = {}
    true_labels  = {}
    for video, pred in zip(video_names, predictions):
        if video not in video_predictions:
            video_predictions[video] = [pred]
        else:
            video_predictions[video].append(pred)
        true_labels [video] = true_value[video_names.index(video)]
        
    # Calculate the total prediction for each video given the threshold
    video_preds = {}

    for video, preds in video_predictions.items():
        avg_pred = np.mean(preds)
        video_preds[video] = int(avg_pred > threshold)
    
    videos = sorted(video_preds.keys())
    y_pred = np.array([video_preds[v] for v in videos])
    y_true = np.array([true_labels[v] for v in videos])
    
    # Return the final predictions and true values as numpy arrays
    return y_true, y_pred



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, 
                            precision_score, recall_score, 
                            accuracy_score, roc_auc_score, f1_score)

def evaluate_video_predictions(y_true, y_pred, class_names=["REAL", "FAKE"], model_name="Model"):

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f"{model_name} - Video-Level Prediction Confusion Matrix")
    plt.show()
    
    # Print metrics
    print("\nClassification Metrics:")
    for name, value in metrics.items():
        print(f"{name.capitalize():<10}: {value:.4f}")
    
    return metrics



from tensorflow.keras import backend as K
import tensorflow as tf
import cv2


def get_grad_cam_dual_input(model, rgb_image, ssim_image, class_idx, layer_name='block_16_project'):
    """Generate Grad-CAM heatmap for dual-input model"""
    # Convert class_idx to integer and verify model output shape
    class_idx = int(class_idx)
    
    # Create gradient model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    # Prepare inputs with proper shapes
    rgb_input = tf.cast(np.expand_dims(rgb_image, axis=0), tf.float32)  # [1, H, W, 3]
    ssim_input = tf.cast(np.expand_dims(ssim_image[..., np.newaxis], axis=0), tf.float32)  # [1, H, W, 1]
    
    with tf.GradientTape() as tape:
        # Get both outputs
        conv_outputs, predictions = grad_model([rgb_input, ssim_input])
        
        # Handle single-output models (binary classification)
        if predictions.shape[-1] == 1:
            loss = predictions[0]  # For binary classification
        else:
            loss = predictions[:, class_idx]  # For multi-class
    
    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)
    
    # Guided Grad-CAM
    cast_conv_outputs = tf.cast(conv_outputs > 0, "float32")
    cast_grads = tf.cast(grads > 0, "float32")
    guided_grads = cast_conv_outputs * cast_grads * grads
    
    # Weight the activation maps
    weights = tf.reduce_mean(guided_grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)
    
    # Process CAM
    cam = np.maximum(cam, 0)
    cam = cam[0]  # Take first (and only) item in batch
    cam = cam / (np.max(cam) + 1e-10)  # Normalize with small epsilon
    
    return cam

import tensorflow as tf
import numpy as np

def get_grad_cam_single_input(model, rgb_image, class_idx, layer_name='block_16_project'):
    """Generate Grad-CAM heatmap for a single-input model (without SSIM)"""
    class_idx = int(class_idx)  # Ensure class index is an integer
    
    # Create a gradient model that outputs feature maps + predictions
    grad_model = tf.keras.models.Model(
        inputs=model.input,  # Single input (only images)
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    # Prepare input
    rgb_input = tf.cast(np.expand_dims(rgb_image, axis=0), tf.float32)  # Shape: [1, H, W, 3]
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(rgb_input)
        
        # Handle binary and multi-class cases
        if predictions.shape[-1] == 1:
            loss = predictions[0]  # Binary classification
        else:
            loss = predictions[:, class_idx]  # Multi-class classification
    
    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)
    
    # Apply Guided Grad-CAM
    cast_conv_outputs = tf.cast(conv_outputs > 0, "float32")
    cast_grads = tf.cast(grads > 0, "float32")
    guided_grads = cast_conv_outputs * cast_grads * grads
    
    # Compute weights for each activation map
    weights = tf.reduce_mean(guided_grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)
    
    # Normalize and process CAM
    cam = np.maximum(cam, 0)  # ReLU operation
    cam = cam[0]  # Remove batch dimension
    cam = cam / (np.max(cam) + 1e-10)  # Normalize to [0, 1] with epsilon
    
    return cam


def overlay_heatmap(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """Overlay heatmap on image"""
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    overlayed = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
    return overlayed
