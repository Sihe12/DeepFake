# DeepFake Detection 
This project implements different deep learning models for detecting deepfakes videos. The dataset utilized is the smaller sample dataset from the [DeepFake Detection Challenge Dataset](https://kaggle.com/deepfake-detection-challenge) from Kaggle. We have implemented the following models:
- **Baseline Model**: A simple CNN model with 4 convolutional layers.
- **Meso-4**: A lightweight model designed for deepfake detection.
- **MesoInception-4.**: An advanced version of Meso-4 with inception modules.
- **MobileNetV3Small**: A lightweight CNN model designed for mobile and edge devices.
- **EfficientNetB0**: A  EfficientNet model pre-trained on ImageNet.
- **Vision Transformer**: A vision transformer model for image classification pre-trained on ImageNet. Specifically, we used the `ViT-B_16` model from keras

## Pipeline
### Data Preprocessing
The data preprocessing pipeline includes the following steps:
1. **Extract frames**: Extract 20 frames from the videos with approximately 0.5 seconds interval.
2. **Extract faces**: Use an MTCNN model to extract faces from the frames.
3. **Resize**: Resize the images to a fixed size of 224x224 pixels.
4. **SSIM**: Calculate the Structural Similarity Index (SSIM) between two consecutive frames. The SSIM maps were stored as separate grayscale images, and the mean and variance of SSIM scores across all frame pairs in a video were computed and stored as metadata.
5. **Save**: Save the face frames, SSIM maps, and metadata in a structured format to make use of Keras’s ImageDataGenerator flow_from_directory API.

### Model Input
The models receives an input on the frame level and not on the video level. Thus, the classification is done on the frame level and then averaged to get the final classification for the video. The models are trained on two different input types:
1. **Single Frame**: Using the extracted face frames as input to the model.
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
- **EfficientNetB0**: An  EfficientNet model pre-trained on ImageNet.
- **Vision Transformer**: A vision transformer model for image classification pre-trained on ImageNet. Specifically, we used the `ViT-B_16` model from keras.

For the multi-model approach:
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
- **Loss Function**: Focal Loss with gamma=2.0 and alpha=0.25. But for ViT, binary cross-entropy is used.
- **Metrics**: Accuracy, Precision, Recall, F1 Score, AUC
- **Early Stopping**: The training is stopped if the validation loss does not improve for 50 epochs.
- **Data Augmentation**: The training data is augmented using the following techniques:
    - **rescale=1./255**: Normalized pixel values to be between 0 and 1.
    - **rotation_range=5**: Applied small rotations to the frame.
    - **width_shift_range=0.03** and **height_shift_range=0.03**: Applied small shifts to the frame.
    - **zoom_range=0.05**: Applied slight zoom.
    - **horizontal_flip=True**: Applied flipping of the frame.
    - **fill_mode=’reflect’**: Used reflection padding to fill the frame after doing the transformations.

### Result Summary
| Model | Parameters | Accuracy | Precision | Recall | F1 Score | AUC |
|-------|------------|----------|-----------|--------|----------|---------|
| Baseline | 423,361 | 0.8649 | 0.9298 | 0.8983 | 0.9138 | 0.8927 |
| Baseline SSIM | 846,337 | 0.7703 | 0.9038 | 0.7966 | 0.8468 | 0.8554 |
| Meso-4 | 24,233 | 0.8919 | 0.9474 | 0.9153 | 0.9310 | 0.9288 |
| Meso-4 SSIM | 447,209 | 0.8649 | 0.9804 | 0.8475 | 0.9091 | 0.9650 |
| MesoInception-4 | 32,089 | 0.8108 | 0.9592 | 0.7966 | 0.8704 | 0.9130 |
| MesoInception-4 SSIM | 455,065 | 0.8378 | 0.9434 | 0.8475 | 0.8929 | 0.9232 |
| MobileNet | 939,697 | 0.4054 | 0.8947 | 0.2881 | 0.4359 | 0.7785 |
| MobileNet SSIM | 1,362,675 | 0.5000 | 0.7895 | 0.5085 | 0.6186 | 0.4475 |
| Efficient | 4,050,852 | 0.4730 | 0.7174 | 0.5593 | 0.6286 | 0.4124 |
| Efficient SSIM | 4,473,830 | 0.7973 | 0.7973 | 1.0000 | 0.8872 | 0.4288 |
| Transformer | 85,799,425 | 0.8243 | 0.8833 | 0.8983 | 0.8908 | 0.8023 |
| Transformer SSIM | 86,222,401 | 0.7162 | 0.8519 | 0.7797 | 0.8142 | 0.7548 |


### Setup
Project used Python 3.10.
We recommend running the experiments on a GPU. For the TensorFlow version we used (2.13) cuDNN 8.6 and CUDA 11.8 is required for GPU support.

1. Clone the repository
3. Install the required packages using the following command:
```bash
pip install -e .
```
