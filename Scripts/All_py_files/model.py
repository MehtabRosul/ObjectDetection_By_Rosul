import numpy as np
from utils import load_model, load_image
from config import MODEL_PATH
import tensorflow as tf

def predict(model, image_path):
    image = load_image(image_path)
    input_tensor = tf.convert_to_tensor([image], dtype=tf.uint8)
    detections = model(input_tensor)
    return detections

def process_detections(detections, threshold=0.5):
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy()

    valid_detections = []
    for i in range(len(scores)):
        if scores[i] > threshold:
            valid_detections.append({
                'box': boxes[i],
                'score': scores[i],
                'class': classes[i]
            })
    return valid_detections
