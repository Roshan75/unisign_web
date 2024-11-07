import tensorflow as tf
import numpy as np

# Load the TFLite classifier model
classifier_interpreter = tf.lite.Interpreter(model_path='models/unisign_main128_30122023_320.tflite')
classifier_interpreter.allocate_tensors()

# Get input and output details
input_details = classifier_interpreter.get_input_details()
output_details = classifier_interpreter.get_output_details()

def classify_image(image):
    """
    Classify the image using a TFLite classifier.
    Parameters:
    - image (numpy array): Preprocessed image.

    Returns:
    - str: Classification result.
    """
    # Prepare input tensor
    image = np.expand_dims(image, axis=0).astype(np.float32)
    classifier_interpreter.set_tensor(input_details[0]['index'], image)
    classifier_interpreter.invoke()
    
    # Get the prediction
    output_data = classifier_interpreter.get_tensor(output_details[0]['index'])
    class_id = np.argmax(output_data[0])  # Example classification
    
    # Map class_id to label (update this mapping according to your model)
    class_labels = {0: "Genuine", 1: "Not Genuine"}
    result = class_labels.get(class_id, "Unknown")
    
    return result
