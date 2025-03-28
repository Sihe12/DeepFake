import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

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


def dual_input_generator(base_gen, ssim_dir, var_mean_dir):
    while True:
        batch_x, batch_y = next(base_gen)
        
        batch_ssim = []
        batch_ssim_stats = []  # Combined mean and variance into one list

        # Get the actual filepaths used for this batch
        current_indices = base_gen.index_array[
            (base_gen.batch_index * base_gen.batch_size) % len(base_gen.index_array):
            ((base_gen.batch_index + 1) * base_gen.batch_size) % len(base_gen.index_array)
        ]

        for i in current_indices:
            img_path = base_gen.filepaths[i]
            rel_path = os.path.relpath(img_path, base_gen.directory)

            # Load SSIM map
            mask_path = os.path.join(ssim_dir, os.path.splitext(rel_path)[0] + '.npy')
            ssim_map = np.load(mask_path)
            ssim_map = cv2.resize(ssim_map, base_gen.target_size)
            batch_ssim.append(ssim_map[..., np.newaxis].astype(np.float32))  

            # Load SSIM mean and variance
            video_name = os.path.basename(img_path).split("_")[0]  # Extract video name
            var_mean_path = os.path.join(var_mean_dir, video_name + '.npy')
            
            if os.path.exists(var_mean_path):
                mean_var_values = np.load(var_mean_path)
                mean_ssim, variance_ssim = mean_var_values[0], mean_var_values[1]
            else:
                mean_ssim, variance_ssim = 0.0, 0.0  # Default values if file is missing
            
            batch_ssim_stats.append([mean_ssim, variance_ssim])  # Combine into (2,) shape

        # Ensure consistent batch size
        if len(batch_x) != len(batch_ssim):
            continue

        # Convert to tensors and return in correct structure
        yield ((tf.convert_to_tensor(batch_x), 
                tf.convert_to_tensor(np.array(batch_ssim)), 
                tf.convert_to_tensor(np.array(batch_ssim_stats), dtype=tf.float32)),  # Now shape (batch_size, 2)
               tf.convert_to_tensor(batch_y))