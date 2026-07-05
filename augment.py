# This Python script is a comprehensive deep learning workflow for training a model using
# TensorFlow/Keras for image classification tasks. Here's a breakdown of what each part of the script
# does:
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tqdm import tqdm

# ----------------- Parameters -----------------
dataset_dir = "color"  # your training folder with subfolders for each class
target_per_class = 1000
img_size = (224, 224)

# Augmentation generator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# ----------------- Process each class -----------------
for class_name in os.listdir(dataset_dir):
    # Skip hidden/system folders
    if class_name.startswith('.') or not os.path.isdir(os.path.join(dataset_dir, class_name)):
        continue

    class_path = os.path.join(dataset_dir, class_name)
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    num_images = len(images)

    print(f"Processing class '{class_name}': {num_images} images")

    if num_images >= target_per_class:
        print(f"  Already >= {target_per_class} images, skipping augmentation.\n")
        continue

    # Calculate how many augmented images needed
    augment_needed = target_per_class - num_images
    print(f"  Need {augment_needed} more images.")

    # Start augmentation loop
    i = 0
    while augment_needed > 0:
        for img_file in tqdm(images, desc=f"Augmenting {class_name}"):
            img_path = os.path.join(class_path, img_file)
            img = load_img(img_path, target_size=img_size)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            # Generate 1 augmented image at a time
            for batch in datagen.flow(x, batch_size=1, save_to_dir=class_path,
                                    save_prefix='aug', save_format='.jpg'):
                i += 1
                augment_needed -= 1
                if augment_needed <= 0:
                    break
            if augment_needed <= 0:
                break

    print(f"  Class '{class_name}' balanced to {target_per_class} images.\n")

print("✅ Dataset balancing complete!")
