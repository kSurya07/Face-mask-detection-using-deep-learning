# Importing necessary libraries
import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

# Path to the input data and annotations
input_data_path = 'input/images'
annotations_path = "input/annotations"
images = [*os.listdir("input/images")]
output_data_path = '.'

# Function to parse annotation object
def parse_annotation_object(annotation_object):
    params = {}
    for param in list(annotation_object):
        if param.tag == 'name':
            params['name'] = param.text
        if param.tag == 'bndbox':
            for coord in list(param):
                params[coord.tag] = int(coord.text)
    return params

# Parsing annotations and creating DataFrame
dataset = []
for anno in glob.glob(annotations_path + "/*.xml"):
    tree = ET.parse(anno)
    root = tree.getroot()
    constants = {'file': root.find('filename').text[0:-4]}
    objects = root.findall('object')
    for obj in objects:
        object_params = parse_annotation_object(obj)
        dataset.append({**constants, **object_params})

df = pd.DataFrame(dataset)


# Removing a test image from the dataset
final_test_image = 'maksssksksss0'
images.remove(f'{final_test_image}.png')
df = df[df["file"] != final_test_image]

# Splitting data into train, test, and validation sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Creating directories for train, test, and validation data
for label in df['name'].unique():
    for d in ['train', 'test', 'val']:
        path = os.path.join(output_data_path, d, label)
        os.makedirs(path, exist_ok=True)

# Function to crop images based on bounding box coordinates
def crop_img(image_path, x_min, y_min, x_max, y_max):
    img = Image.open(image_path)
    cropped = img.crop((x_min - (x_max - x_min) * 0.1, y_min - (y_max - y_min) * 0.1, x_max + (x_max - x_min) * 0.1, y_max + (y_max - y_min) * 0.1))
    return cropped

# Saving images to directories
def save_image(image, image_name, dataset_type, label):
    output_path = os.path.join(output_data_path, dataset_type, label, f'{image_name}.png')
    image.save(output_path)

for dataset, dataset_type in [(train_df, 'train'), (test_df, 'test'), (val_df, 'val')]:
    for _, row in dataset.iterrows():
        image_path = os.path.join(input_data_path, row['file'] + '.png')
        image = crop_img(image_path, row['xmin'], row['ymin'], row['xmax'], row['ymax'])
        save_image(image, row['file'] + '_' + str((row['xmin'], row['ymin'])), dataset_type, row['name'])

# Creating a CNN model
model = Sequential([
    Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', input_shape=(35, 35, 3)),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),
    Dropout(0.3),
    Flatten(),
    Dense(units=500, activation='relu'),
    Dropout(0.3),
    Dense(units=3, activation='softmax')
])

# Compiling the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])


# Data augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255, horizontal_flip=True, zoom_range=0.1, shear_range=0.2, width_shift_range=0.1,
    height_shift_range=0.1, rotation_range=4, vertical_flip=False
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Generating data from directories
batch_size = 8
train_generator = datagen.flow_from_directory(directory='train', target_size=(35, 35),
                                              class_mode="categorical", batch_size=batch_size, shuffle=True)

val_generator = val_datagen.flow_from_directory(directory='val', target_size=(35, 35),
                                                class_mode="categorical", batch_size=batch_size, shuffle=True)

test_generator = val_datagen.flow_from_directory(directory='test', target_size=(35, 35),
                                                 class_mode="categorical", batch_size=batch_size, shuffle=False)

# Training the model
model.fit(train_generator, epochs=50, validation_data=val_generator, callbacks=[])

# Evaluating the model
model_loss, model_acc = model.evaluate(test_generator)
print(f'Test Loss: {model_loss}, Test Accuracy: {model_acc}')