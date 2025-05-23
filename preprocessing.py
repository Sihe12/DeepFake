import tensorflow as tf
import os

gpu = True
if gpu:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print(os.environ['CUDA_VISIBLE_DEVICES'])  
    
    
import random
import numpy as np

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
from skimage.metrics import structural_similarity as ssim

data_folder = "data"
output_folder = "data_images"

if os.path.exists(output_folder) and len(os.listdir(output_folder)) > 0:
    print("Skipping processing: 'data_images' folder is already populated.")
else:
    os.makedirs(output_folder, exist_ok=True)
    
    detector = MTCNN()
    
    num_frames = 20
    frame_interval = 15  
    max_consecutive_checks = 14  
    frame_size = (224, 224)
    
    with open("data/metadata.json", "r") as f:
        video_labels = json.load(f)
    
    image_labels = {}
    
    for video_name in os.listdir(data_folder):
        video_path = os.path.join(data_folder, video_name)
        
        if video_name.endswith(".mp4"):
            reader = imageio.get_reader(video_path, "ffmpeg")
            num_total_frames = reader.count_frames()
            face_count = 0  
            prev_face = None  
            faces_to_save = []  
            
            i = 0
            while i < num_total_frames and face_count < num_frames:
                found = False
                j = 0  

                
                while j < max_consecutive_checks and (i + j) < num_total_frames:
                    frame = reader.get_data(i + j)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                    
                    faces = detector.detect_faces(frame_rgb, 
                                                  min_face_size =10,     
                                                  threshold_pnet=0.5,  
                                                  threshold_rnet=0.6,  
                                                  threshold_onet=0.7  
                                                  )
                    if faces:
                        found = True
                        i += j  
                        break  
                    j += 1  

                if not found:
                    i += frame_interval  
                    continue  

                
                best_face = None
                best_ssim = -1  
                for face in faces:
                    x, y, w, h = face['box']
                    face_crop = frame[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_crop, frame_size)

                    if prev_face is not None:
                        prev_gray = cv2.cvtColor(prev_face, cv2.COLOR_BGR2GRAY)
                        curr_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

                        similarity = ssim(prev_gray, curr_gray)

                        if similarity > best_ssim:
                            best_ssim = similarity
                            best_face = face_resized
                    else:
                        best_face = face_resized  

                prev_face = best_face  
                
                faces_to_save.append((best_face, f"{video_name.split('.')[0]}_{i}.jpg"))
                face_count += 1

                i += frame_interval - j  
            
            reader.close()
            
            if face_count < num_frames:
                print(f"Skipping video: {video_name} - Less than {num_frames} faces detected.")
                continue
            
            for face_resized, image_name in faces_to_save:
                image_path = os.path.join(output_folder, image_name)
                cv2.imwrite(image_path, face_resized)
                video_metadata = video_labels.get(video_name)
                if video_metadata:
                    image_labels[image_name] = video_metadata["label"]
            
            print(f"Done with video {video_name}")
    
    with open("image_labels.json", "w") as f:
        json.dump(image_labels, f, indent=4)
    
    print("Processing complete. Images saved in 'data_images', labels in 'image_labels.json'.")



import matplotlib.pyplot as plt

image_labels = json.load(open("image_labels.json", "r"))

movie_labels = {}

for filename, label in image_labels.items():
    movie_name = filename.split("_")[0]  
    movie_labels[movie_name] = label  
    
real_movies_count = sum(1 for label in movie_labels.values() if label == "REAL")
fake_movies_count = sum(1 for label in movie_labels.values() if label == "FAKE")

labels = ['REAL', 'FAKE']
counts = [real_movies_count, fake_movies_count]

plt.bar(labels, counts, color=['green', 'red'])
plt.xlabel('Video Type')
plt.ylabel('Count')
plt.title('REAL vs FAKE Video Counts')
plt.show()



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



train_movies = set(train[:, 0])
val_movies = set(val[:, 0])
test_movies = set(test[:, 0])

def get_images_from_movies(movie_set, image_labels):
    return np.array([[img, label] for img, label in image_labels.items() if img.split("_")[0] in movie_set], dtype=object)

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

plt.show()



from PIL import Image

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
    save_images(train_images, 'train')
    save_images(val_images, 'val')
    save_images(test_images, 'test')

def precompute_intra_video_ssim_masks(dataset_dir, save_dir, var_mean_save_dir):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(var_mean_save_dir, exist_ok=True)
        
    for label in os.listdir(dataset_dir):
        label_dir = os.path.join(dataset_dir, label)
        label_save_dir = os.path.join(save_dir, label)
        label_var_mean_dir = os.path.join(var_mean_save_dir, label)

        os.makedirs(label_save_dir, exist_ok=True)  
        os.makedirs(label_var_mean_dir, exist_ok=True)  

        video_frames = {}
        for img_file in os.listdir(label_dir):
            video_name = img_file.split("_")[0]  
            if video_name not in video_frames:
                video_frames[video_name] = []
            video_frames[video_name].append(img_file)

        for video in video_frames:
            video_frames[video] = sorted(video_frames[video])

        for video_name, frames in video_frames.items():
            num_frames = len(frames)

            ssim_scores = [] 

            for i in range(num_frames):
                img_path = os.path.join(label_dir, frames[i])
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                img = cv2.resize(img, (224, 224))

                if i == 0:
                    next_frame_path = os.path.join(label_dir, frames[i + 1])
                    next_frame = cv2.imread(next_frame_path, cv2.IMREAD_GRAYSCALE)
 
                    next_frame = cv2.resize(next_frame, (224, 224))
                    score, ssim_map = ssim(img, next_frame, full=True)
                else:
                    prev_frame_path = os.path.join(label_dir, frames[i - 1])
                    prev_frame = cv2.imread(prev_frame_path, cv2.IMREAD_GRAYSCALE)
                    prev_frame = cv2.resize(prev_frame, (224, 224))
                    score, ssim_map = ssim(img, prev_frame, full=True)

                ssim_scores.append(score) 

                ssim_map = (ssim_map - ssim_map.min()) / (ssim_map.max() - ssim_map.min())

                mask_path = os.path.join(label_save_dir, f"{os.path.splitext(frames[i])[0]}.npy")
                np.save(mask_path, ssim_map.astype(np.float32))
                
                if ssim_scores:
                    mean_ssim = np.mean(ssim_scores)
                    variance_ssim = np.var(ssim_scores)
                    var_mean_path = os.path.join(label_var_mean_dir, f"{video_name}.npy")
                    np.save(var_mean_path, np.array([mean_ssim, variance_ssim], dtype=np.float32))
            
                
if all(os.path.exists(folder) for folder in ['train_ssim', 'val_ssim', 'test_ssim']) and \
   all(os.path.exists(folder) for folder in ['train_ssim_var_mean', 'val_ssim_var_mean', 'test_ssim_var_mean']):
    print("Skipping processing: 'train_ssim, val_ssim, test_ssim, train_ssim_var_mean, val_ssim_var_mean, test_ssim_var_mean' folders already exist.")
else:
    precompute_intra_video_ssim_masks("train", 'train_ssim', 'train_ssim_var_mean')
    precompute_intra_video_ssim_masks("val", 'val_ssim', 'val_ssim_var_mean')
    precompute_intra_video_ssim_masks("test", 'test_ssim', 'test_ssim_var_mean')