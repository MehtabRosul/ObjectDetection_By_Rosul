import tensorflow as tf
from config import MODEL_PATH

def load_model():
    """
    Loads the TensorFlow model from the specified MODEL_PATH.
    """
    try:
        model = tf.saved_model.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}: {e}")
        raise

def process_annotations(annotations_file):
    # Add your processing logic here
    pass


# Example utility function
def preprocess_image(image_path):
    """
    Preprocesses an image before feeding it to the model.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, [640, 640])
    image = tf.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize the image
    return image

def predict(model, image):
    """
    Runs inference on a single image using the loaded model.
    """
    preprocessed_image = preprocess_image(image)
    predictions = model(preprocessed_image)
    return predictions
