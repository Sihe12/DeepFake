import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

def get_video_prediction(predictions, threshold=0.5, test_generator=None):
    image_filenames = test_generator.filenames  
    video_names = [os.path.splitext(os.path.basename(f))[0].split('_')[0] for f in image_filenames]
    

    true_value = test_generator.classes  
    video_predictions = {}
    true_labels  = {}
    for video, pred in zip(video_names, predictions):
        if video not in video_predictions:
            video_predictions[video] = [pred]
        else:
            video_predictions[video].append(pred)
        true_labels [video] = true_value[video_names.index(video)]
        
    video_preds_raw = {}

    video_preds = {}

    for video, preds in video_predictions.items():
        avg_pred = np.mean(preds)
        video_preds[video] = int(avg_pred > threshold)
        video_preds_raw[video] = avg_pred
    
    videos = sorted(video_preds.keys())
    y_pred_probs = np.array([video_preds_raw[v] for v in videos])
    y_pred_binary = np.array([video_preds[v] for v in videos])
    y_true = np.array([true_labels[v] for v in videos])
    
    return y_true, y_pred_binary, y_pred_probs



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, 
                            precision_score, recall_score, 
                            accuracy_score, roc_auc_score, f1_score)

def evaluate_video_predictions(y_true, y_pred_probs, y_pred_binary, class_names=["REAL", "FAKE"], model_name="Model"):

    from sklearn.metrics import roc_curve

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'precision': precision_score(y_true, y_pred_binary),
        'recall': recall_score(y_true, y_pred_binary),
        'f1': f1_score(y_true, y_pred_binary),
        'auc_roc': roc_auc_score(y_true, y_pred_probs)  
    }
    
    cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])

    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f"{model_name} - Video-Level Prediction Confusion Matrix")
    plt.show()

    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC Curve (AUC = {metrics['auc_roc']:.4f})")
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--")  
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} - ROC Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    print("\nClassification Metrics:")
    for name, value in metrics.items():
        print(f"{name.capitalize():<10}: {value:.4f}")

    return metrics


def dual_input_generator(base_gen, ssim_dir, var_mean_dir):
    while True:
        batch_x, batch_y = next(base_gen)
        
        batch_ssim = []
        batch_ssim_stats = []  
        current_indices = base_gen.index_array[
            (base_gen.batch_index * base_gen.batch_size) % len(base_gen.index_array):
            ((base_gen.batch_index + 1) * base_gen.batch_size) % len(base_gen.index_array)
        ]

        for i in current_indices:
            img_path = base_gen.filepaths[i]
            rel_path = os.path.relpath(img_path, base_gen.directory)

            mask_path = os.path.join(ssim_dir, os.path.splitext(rel_path)[0] + '.npy')
            ssim_map = np.load(mask_path)
            ssim_map = cv2.resize(ssim_map, base_gen.target_size)
            batch_ssim.append(ssim_map[..., np.newaxis].astype(np.float32))  

            video_name = os.path.basename(img_path).split("_")[0] 
            var_mean_path = os.path.join(var_mean_dir, video_name + '.npy')
            
            if os.path.exists(var_mean_path):
                mean_var_values = np.load(var_mean_path)
                mean_ssim, variance_ssim = mean_var_values[0], mean_var_values[1]
            else:
                mean_ssim, variance_ssim = 0.0, 0.0  
            
            batch_ssim_stats.append([mean_ssim, variance_ssim])  

        if len(batch_x) != len(batch_ssim):
            continue

        yield ((tf.convert_to_tensor(batch_x), 
                tf.convert_to_tensor(np.array(batch_ssim)), 
                tf.convert_to_tensor(np.array(batch_ssim_stats), dtype=tf.float32)), 
                tf.convert_to_tensor(batch_y))
        
        
import tensorflow.keras.backend as K


def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)  
        loss = -y_true * alpha * K.pow(1 - y_pred, gamma) * K.log(y_pred) - (1 - y_true) * (1 - alpha) * K.pow(y_pred, gamma) * K.log(1 - y_pred)
        return K.mean(loss)
    return loss