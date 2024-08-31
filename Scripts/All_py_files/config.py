import os

# Model and checkpoint paths
MODEL_PATH = os.path.join('C:', os.sep, 'Python', 'Scripts', 'Project', 'ObjectDetection', 'venv', 'Pretrained_model', 'ssd_mobilenet_v2', 'efficientdet_d3_coco17_tpu-32', 'saved_model')
CHECKPOINT_PATH = os.path.join('C:', os.sep, 'Python', 'Scripts', 'Project', 'ObjectDetection', 'venv', 'Pretrained_model', 'ssd_mobilenet_v2', 'efficientdet_d3_coco17_tpu-32', 'checkpoint')

# Dataset paths
ANNOTATIONS_PATH = os.path.join('C:', os.sep, 'Python', 'Scripts', 'Project', 'ObjectDetection', 'venv', 'Datasets', 'annotations')
IMAGES_PATH = os.path.join('C:', os.sep, 'Python', 'Scripts', 'Project', 'ObjectDetection', 'venv', 'Datasets', 'val2014')

# Annotation files
ANNOTATIONS_FILES = {
    'captions_train': os.path.join(ANNOTATIONS_PATH, 'captions_train2014.json'),
    'captions_val': os.path.join(ANNOTATIONS_PATH, 'captions_val2014.json'),
    'instances_train': os.path.join(ANNOTATIONS_PATH, 'instances_train2014.json'),
    'instances_val': os.path.join(ANNOTATIONS_PATH, 'instances_val2014.json'),
    'person_keypoints_train': os.path.join(ANNOTATIONS_PATH, 'person_keypoints_train2014.json'),
    'person_keypoints_val': os.path.join(ANNOTATIONS_PATH, 'person_keypoints_val2014.json'),
}

# Additional paths needed
VAL_IMAGES_DIR = IMAGES_PATH  # Assuming this is where the validation images are stored
OUTPUT_DIR = os.path.join('C:', os.sep, 'Python', 'Scripts', 'Project', 'ObjectDetection', 'venv', 'Outputs')  # Path where output images or results will be stored

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
