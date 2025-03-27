import tensorflow as tf
import os

gpu = False
# Use gpu if available
if gpu:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print(os.environ['CUDA_VISIBLE_DEVICES'])  # Check the value
    
    
import random
import numpy as np
# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)





import cv2
import imageio
import os
import random
import json
from mtcnn import MTCNN

# Paths
data_folder = "data"
output_folder = "data_images"

# If the output folder is not empty, skip processing
if os.path.exists(output_folder) and len(os.listdir(output_folder)) > 0:
    print("Skipping processing: 'data_images' folder is already populated.")
else:
    os.makedirs(output_folder, exist_ok=True)


    # Initialize MTCNN
    detector = MTCNN()

    # Parameters
    num_frames = 20
    frame_size = (224, 224)

    # Load labels from the existing metadata
    with open("data/metadata.json", "r") as f:
        video_labels = json.load(f)

    # Dictionary to store image-label mappings
    image_labels = {}

    # Process videos
    for video_name in os.listdir(data_folder):
        video_path = os.path.join(data_folder, video_name)

        if video_name.endswith(".mp4"):
            reader = imageio.get_reader(video_path, "ffmpeg")
            total_frames = reader.count_frames()

            # Select frame indices
            selected_frame_indices = sorted(random.sample(range(total_frames), num_frames))

            # Extract frames and process
            for i, frame in enumerate(reader):
                if i in selected_frame_indices:
                    frame_rgb = frame
                    if frame.ndim == 2:  # if grayscale
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                    elif frame.shape[2] == 4:  # if RGBA
                        frame_rgb = frame[:, :, :3]
                    # Detect faces in the frame and choose the first face
                    faces = detector.detect_faces(frame)
                    
                    # If no faces are found, skip this frame
                    if len(faces) == 0:
                        continue

                    # Get the bounding box of the first face
                    face = faces[0]
                    x, y, w, h = face['box']

                    # Crop the face region
                    face_crop = frame[y:y+h, x:x+w]
                    
                    if face_crop.size == 0:  # Ensure the cropped face is valid
                        continue
                    
                    # Resize the cropped face to 224x224
                    face_resized = cv2.resize(face_crop, frame_size)

                    # Generate a unique image name for the face
                    image_name = f"{video_name.split('.')[0]}_{i}.jpg"
                    image_path = os.path.join(output_folder, image_name)

                    # Save the resized face image
                    cv2.imwrite(image_path, face_resized)

                    # Store the label mapping
                    video_metadata = video_labels.get(video_name)
                    if video_metadata:
                        image_labels[image_name] = video_metadata["label"]

            reader.close()

    # Save the label mappings to a JSON file
    with open("image_labels.json", "w") as f:
        json.dump(image_labels, f, indent=4)

    print("Processing complete. Images saved in 'data_images', labels in 'image_labels.json'.")



import matplotlib.pyplot as plt

image_labels = json.load(open("image_labels.json", "r"))

movie_labels = {}

for filename, label in image_labels.items():
    movie_name = filename.split("_")[0]  # Extract movie name
    movie_labels[movie_name] = label  # Store only one label per movie
    
# Count the number of unique REAL and FAKE videos
real_movies_count = sum(1 for label in movie_labels.values() if label == "REAL")
fake_movies_count = sum(1 for label in movie_labels.values() if label == "FAKE")

# Plotting the results
labels = ['REAL', 'FAKE']
counts = [real_movies_count, fake_movies_count]

plt.bar(labels, counts, color=['green', 'red'])
plt.xlabel('Video Type')
plt.ylabel('Count')
plt.title('REAL vs FAKE Video Counts')
plt.show()


# load the images and labels and set it to a tf.data.Dataset

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

fakes = np.empty((0, 2), dtype=object)
reals = np.empty((0, 2), dtype=object)

for image, label in movie_labels.items():
    if label == "FAKE":
        
        fakes = np.append(fakes, np.array([[image, label]]), axis=0)
    else:
        reals = np.append(reals, np.array([[image, label]]), axis=0)
        
data = np.vstack((fakes, reals))


from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.2, random_state=SEED, stratify=data[:, 1])

train, val = train_test_split(train, test_size=0.2, random_state=SEED, stratify=train[:, 1])



# Convert to sets for quick lookup
train_movies = set(train[:, 0])
val_movies = set(val[:, 0])
test_movies = set(test[:, 0])

# Function to filter images based on movie set
def get_images_from_movies(movie_set, image_labels):
    return np.array([[img, label] for img, label in image_labels.items() if img.split("_")[0] in movie_set], dtype=object)

# Filter images for each dataset
train_images = get_images_from_movies(train_movies, image_labels)
val_images = get_images_from_movies(val_movies, image_labels)
test_images = get_images_from_movies(test_movies, image_labels)



fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

unique_labels, label_counts = np.unique(train_images[:, 1], return_counts=True)
axes[0].bar(unique_labels, label_counts, edgecolor='black')
axes[0].set_xticks(unique_labels)
axes[0].set_xticklabels(['Fakes', 'Real'])
axes[0].set_xlabel('Label')
axes[0].set_ylabel('Number of Images')
axes[0].set_title('Number of Images per Label in Training Set')

unique_labels, label_counts = np.unique(val_images[:, 1], return_counts=True)
axes[1].bar(unique_labels, label_counts, edgecolor='black')
axes[1].set_xticks(unique_labels)
axes[0].set_xticklabels(['Fakes', 'Real'])
axes[1].set_xlabel('Label')
axes[1].set_ylabel('Number of Images')
axes[1].set_title('Number of Images per Label in Validation Set')

unique_labels, label_counts = np.unique(test_images[:, 1], return_counts=True)
axes[2].bar(unique_labels, label_counts, edgecolor='black')
axes[2].set_xticks(unique_labels)
axes[0].set_xticklabels(['Fakes', 'Real'])
axes[2].set_xlabel('Label')
axes[2].set_ylabel('Number of Images')
axes[2].set_title('Number of Images per Label in Test Set')

# Display the plot
plt.show()



from PIL import Image

# Mapping labels
label_mapping = {"REAL": 0, "FAKE": 1}


def save_images(data, base_folder):
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    for img_path, label in data:
        # Convert label to 0 or 1
        numeric_label = label_mapping[label]
        
        label_folder = os.path.join(base_folder, str(numeric_label))
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)
        img = Image.open("data_images/" + img_path)
        img.save(os.path.join(label_folder, os.path.basename(img_path)))

if os.path.exists('train') and os.path.exists('val') and os.path.exists('test') and len(os.listdir(output_folder)) > 0:
    print("Skipping processing: 'train, val, test' folder is already full.")
else:
    # Save the images for each dataset
    save_images(train_images, 'train')
    save_images(val_images, 'val')
    save_images(test_images, 'test')
    
    
    
def SSIM_Avg():    
    # --- Calculate average real image for SSIM ---
    real_train_dir = os.path.join('train', '0')  # Assuming '0' is REAL class
    real_image_files = [os.path.join(real_train_dir, f) for f in os.listdir(real_train_dir)]

    # Compute average of all real training images
    avg_real_image = np.zeros((224, 224), dtype=np.float32)
    for img_file in real_image_files:
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (224, 224))
            avg_real_image += img.astype(np.float32)
    avg_real_image /= len(real_image_files)
    avg_real_image = avg_real_image.astype(np.uint8)
    return avg_real_image



from skimage.metrics import structural_similarity as ssim

def precompute_ssim_masks(dataset_dir, save_dir, avg_real_image):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for label in os.listdir(dataset_dir):
        label_dir = os.path.join(dataset_dir, label)
        label_save_dir = os.path.join(save_dir, label)
        os.makedirs(label_save_dir, exist_ok=True)  # Create label directory

        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                img = cv2.resize(img, (224, 224))

                # Compute SSIM map
                _, ssim_map = ssim(img, avg_real_image, full=True)

               # Normalize to [0,1] and save as float32 numpy array
                ssim_map = (ssim_map - ssim_map.min()) / (ssim_map.max() - ssim_map.min())
                mask_path = os.path.join(label_save_dir, f"{os.path.splitext(img_file)[0]}.npy")
                np.save(mask_path, ssim_map.astype(np.float32))
                
# Precompute SSIM masks for each dataset
if os.path.exists('train_ssim') and os.path.exists('val_ssim') and os.path.exists('test_ssim'):
    print("Skipping processing: 'train_ssim, val_ssim, test_ssim' folder is already full.")
else:
    # Save the images for each dataset
    avg_real_image = SSIM_Avg()
    precompute_ssim_masks("train", 'train_ssim', avg_real_image)
    precompute_ssim_masks("val", 'val_ssim', avg_real_image)
    precompute_ssim_masks("test", 'test_ssim', avg_real_image)
