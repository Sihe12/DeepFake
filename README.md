# DeepFake Detection 
This project implements different deep learning models for detecting deepfakes videos. The dataset utilized is the smaller sample dataset from the [DeepFake Detection Challenge Dataset](https://kaggle.com/deepfake-detection-challenge) from Kaggle. We have implemented the following models:
- **Baseline Model**: A simple CNN model with 4 convolutional layers.
- **MesoNet 4**: A lightweight model designed for deepfake detection.
- **MesoInception-4.**: An advanced version of MesoNet with inception modules.
- **MobileNetV3Small**: A lightweight CNN model designed for mobile and edge devices.
- **EfficientNetB0**: A  EfficientNet model pre-trained on ImageNet.
- **Vision Transformer**: A vision transformer model for image classification pre-trained on ImageNet. Specifically, we used the `ViT-B_16` model from keras

## Pipeline
### Data Preprocessing
The data preprocessing pipeline includes the following steps:
1. **Extract frames**: Extract 20 frames from the videos with approximately 0.5 seconds interval.
2. **Extract faces**: Use a MTCNN model to extract faces from the frames.
3. **Resize**: Resize the images to a fixed size of 224x224 pixels.
4. **SSIM**: Calculate the Structural Similarity Index (SSIM) between two consecutive frames. The SSIM maps were stored as separate grayscale images, and the mean and variance of SSIM scores across all frame pairs in a video were computed and stored as metadata.
5. **Save**: Save the face frames, SSIM maps, and metadata in a structured format to make use of Keras’s ImageDataGenerator flow_from_directory API.

### Model Input
The models receives an input on the frame level and not on the video level. Thus the classification is done on the frame level and then averaged to get the final classification for the video. The models are trained on two different input types:
1. **Signle Frame**: Using the extracted face frames as input to the model.
2. **Multi Model**: The model receives three separate inputs per frame:
    – The face frame.
    – The corresponding SSIM map as a grayscale image.
    – Two scalar values representing the SSIM mean and variance.
    These inputs are processed through separate model branches and then concatenated and passed to the final classification layer.

### Models
We have implemented the following models:
- **Baseline Model**: A simple CNN model with 4 convolutional layers.
- **MesoNet 4**: A lightweight model designed for deepfake detection.
- **MesoInception-4.**: An advanced version of MesoNet with inception modules.
- **MobileNetV3Small**: A lightweight CNN model designed for mobile and edge devices.
- **EfficientNetB0**: A  EfficientNet model pre-trained on ImageNet.
- **Vision Transformer**: A vision transformer model for image classification pre-trained on ImageNet. Specifically, we used the `ViT-B_16` model from keras.

For the multi model approach:
- **Main Model**: The main model is the different models mentioned above.
- **SSIM Model**: A simple CNN model identical to the Baseline model. 
- **SSIM Statistics**: A small dense model.
These are concatenated and passed to the final classification layer.

### Model Training
The models are trained using the following parameters:
- **Batch Size**: 16
- **Epochs**: 200
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss Function**: Focal Loss with gamma=2.0 and alpha=0.25. But for ViT binary cross entropy is used.
- **Metrics**: Accuracy, Precision, Recall, F1 Score, AUC
- **Early Stopping**: The training is stopped if the validation loss does not improve for 50 epochs.

# Base Model
Classification Metrics:<br>
Accuracy  : 0.8649<br>
Precision : 0.9298<br>
Recall    : 0.8983<br>
F1        : 0.9138<br>
Auc_roc   : 0.8927
Parameters: 423,361

# Base Model SSIM
Classification Metrics:<br>
Accuracy  : 0.7703<br>
Precision : 0.9038<br>
Recall    : 0.7966<br>
F1        : 0.8468<br>
Auc_roc   : 0.8554
Parameters: 846,337
![alt text](base:model_SSIM.png)

# Efficient
Classification Metrics:<br>
Accuracy  : 0.4730<br>
Precision : 0.7174<br>
Recall    : 0.5593<br>
F1        : 0.6286<br>
Auc_roc   : 0.4124
Parameters: 4,050,852

# Efficient SSIM
Classification Metrics:<br>
Accuracy  : 0.7973<br>
Precision : 0.7973<br>
Recall    : 1.0000<br>
F1        : 0.8872<br>
Auc_roc   : 0.4288
Parameters: 4,473,830

# Mesonet
Classification Metrics:<br>
Accuracy  : 0.8919<br>
Precision : 0.9474<br>
Recall    : 0.9153<br>
F1        : 0.9310<br>
Auc_roc   : 0.9288
Parameters: 24,233

# Mesonet SSIM
Classification Metrics:<br>
Accuracy  : 0.8378<br>
Precision : 0.9273<br>
Recall    : 0.8644<br>
F1        : 0.8947<br>
Auc_roc   : 0.9141
Parameters: 53,473 

# MesoNet-Inception
Classification Metrics:<br>
Accuracy  : 0.8108<br>
Precision : 0.9592<br>
Recall    : 0.7966<br>
F1        : 0.8704<br>
Auc_roc   : 0.9130
Parameters: 32,089 

# MesoNet-Inception SSIM
Classification Metrics:<br>
Accuracy  : 0.8919<br>
Precision : 0.9474<br>
Recall    : 0.9153<br>
F1        : 0.9310<br>
Auc_roc   : 0.9220
Parameters: 61,329

# Transformer
Classification Metrics:<br>
Accuracy  : 0.7973<br>
Precision : 0.7973<br>
Recall    : 1.0000<br>
F1        : 0.8872<br>
Auc_roc   : 0.5000<br>
Parameters: 85,799,425

# Transformer SSIM
Classification Metrics:<br>
Accuracy  : 0.7973<br>
Precision : 0.7973<br>
Recall    : 1.0000<br>
F1        : 0.8872<br>
Auc_roc   : 0.5000<br>
Parameters: 86,222,401

# MobileNet
Classification Metrics:
Accuracy  : 0.4054
Precision : 0.8947
Recall    : 0.2881
F1        : 0.4359
Auc_roc   : 0.7785
Parameters: 939,697

# MobileNet SSIM
Classification Metrics:
Accuracy  : 0.5000
Precision : 0.7895
Recall    : 0.5085
F1        : 0.6186
Auc_roc   : 0.4475


### Result Summary
| Model | Parameters | Accuracy | Precision | Recall | F1 Score | AUC ROC |
|-------|------------|----------|-----------|--------|----------|---------|



### Setup
1. Clone the repository
2. Install the required packages using the following command:
```bash
pip install -e .
```