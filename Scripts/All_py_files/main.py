import tensorflow as tf
import cv2
import numpy as np
import os
from utils import load_model, process_annotations
from config import VAL_IMAGES_DIR, OUTPUT_DIR, ANNOTATIONS_FILES

# Set TensorFlow logger to only show errors
tf.get_logger().setLevel('ERROR')

print(f"Running script from: {os.path.abspath(__file__)}")

def draw_detections(frame, detections):
    # Unpack detections
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)
    detection_scores = detections['detection_scores'][0].numpy()

    height, width, _ = frame.shape
    for i in range(detection_boxes.shape[0]):
        if detection_scores[i] < 0.5:  # Only consider high confidence detections
            continue

        class_id = detection_classes[i]  # Ensure this is a scalar
        box = detection_boxes[i] * np.array([height, width, height, width])
        ymin, xmin, ymax, xmax = box.astype(int)

        # Draw the bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        label = f"Class {class_id}: {detection_scores[i]:.2f}"
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def preprocess_frame(frame):
    # Resize and normalize the frame for the model
    resized_frame = cv2.resize(frame, (224, 224))  # Reduce frame size
    input_frame = np.expand_dims(resized_frame, axis=0)
    input_frame = input_frame.astype(np.uint8)  # Model expects uint8
    return input_frame

def main():
    model = load_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        input_frame = preprocess_frame(frame)

        # Perform detection
        detections = model(input_frame)

        # Draw detections on the frame
        draw_detections(frame, detections)

        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
