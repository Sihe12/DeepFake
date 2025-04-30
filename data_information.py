import json

with open('data/metadata.json', 'r') as file:
    metadata = json.load(file)

total_videos = len(metadata)
real_count = 0
fake_count = 0

for video, details in metadata.items():
    if details['label'] == 'REAL':
        real_count += 1
    elif details['label'] == 'FAKE':
        fake_count += 1


print(f'Total number of videos: {total_videos}')
print(f'Number of REAL videos: {real_count}')
print(f'Number of FAKE videos: {fake_count}')


print("After pre processing:")

with open('image_labels.json', 'r') as file:
    metadata = json.load(file)

total_videos = 0
real_count = 0
fake_count = 0

videos = {
}

for image, label in metadata.items():
    base_name = image.split('_')[0] + '.jpg'
    if base_name not in videos:
        videos[base_name] = label
        total_videos += 1
        if label == 'FAKE':
            fake_count += 1
        elif label == 'REAL':
            real_count += 1
        

print(f'Total number of videos: {total_videos}')
print(f'Number of REAL videos: {real_count}')
print(f'Number of FAKE vidoes: {fake_count}')
