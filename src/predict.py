import tensorflow as tf
import cv2
import numpy as np

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def count_eggs(image_path, model):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    return int(round(prediction[0][0]))

def process_video(video_path, model, output_path, frame_interval=1):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(rgb_frame, (224, 224))
            normalized_frame = resized_frame / 255.0
            input_frame = np.expand_dims(normalized_frame, axis=0)
            
            egg_count = int(round(model.predict(input_frame)[0][0]))
            
            cv2.putText(frame, f"Eggs: {egg_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()

if __name__ == "__main__":
    model_path = 'egg_counter_model.h5'
    model = load_model(model_path)
    
    # For single image prediction
    image_path = 'path/to/test/image.jpg'
    egg_count = count_eggs(image_path, model)
    print(f"Number of eggs detected: {egg_count}")
    
    # For video processing
    video_path = 'path/to/input/video.mp4'
    output_path = 'path/to/output/video.mp4'
    process_video(video_path, model, output_path)