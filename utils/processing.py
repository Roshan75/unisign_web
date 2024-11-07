import cv2

def preprocess_image(image):
    """
    Preprocess the image for the model pipeline.
    Parameters:
    - image (numpy array): Original image in BGR format.

    Returns:
    - numpy array: Preprocessed image.
    """
    # Resize image if required by the model (e.g., 224x224 for classifiers)
    processed_image = cv2.resize(image, (224, 224))
    # Normalize image to [0, 1] range
    processed_image = processed_image / 255.0
    # Convert to RGB if needed (some models require RGB format)
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    return processed_image


def resize_with_padding(image, target_size=720):
    # Get original dimensions
    h, w = image.shape[:2]

    # Determine the scale to maintain aspect ratio
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize the image
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Calculate padding to add to each side
    top_pad = (target_size - new_h) // 2
    bottom_pad = target_size - new_h - top_pad
    left_pad = (target_size - new_w) // 2
    right_pad = target_size - new_w - left_pad

    # Pad the image with white (255, 255, 255)
    padded_image = cv2.copyMakeBorder(resized_image, top_pad, bottom_pad, left_pad, right_pad,
                                      cv2.BORDER_CONSTANT, value=[255, 255, 255])

    return padded_image