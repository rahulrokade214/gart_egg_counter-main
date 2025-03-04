import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize pixel values
    return img

def augment_data(image):
    # Apply random rotation
    angle = np.random.uniform(-20, 20)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    image = cv2.warpAffine(image, M, (w, h))
    
    # Apply random brightness adjustment
    brightness = np.random.uniform(0.8, 1.2)
    image = image * brightness
    image = np.clip(image, 0, 1)
    
    return image

def prepare_dataset(data_dir, target_size=(224, 224)):
    images = []
    labels = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(data_dir, filename)
            img = load_and_preprocess_image(img_path, target_size)
            images.append(img)
            
            # Extract label from filename (assuming format: image_label.jpg)
            label = int(filename.split('_')[1].split('.')[0])
            labels.append(label)
    
    return np.array(images), np.array(labels)

def split_dataset(images, labels, test_size=0.2, val_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    data_dir = "data/images"
    images, labels = prepare_dataset(data_dir)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(images, labels)
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")