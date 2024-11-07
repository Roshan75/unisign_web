# import tensorflow as tf
# import numpy as np
#
# # Load the TFLite object detection model
# detector_interpreter = tf.lite.Interpreter(model_path='models/unisign_graygrid128_08122023_oneplus64.tflite')
# detector_interpreter.allocate_tensors()
#
# # Get input and output details
# detector_input_details = detector_interpreter.get_input_details()
# detector_output_details = detector_interpreter.get_output_details()
#
# def detect_objects(image):
#     """
#     Detect objects in the image using a TFLite YOLO model.
#     Parameters:
#     - image (numpy array): Preprocessed image.
#
#     Returns:
#     - list: List of detected objects with labels and confidence scores.
#     """
#     image = np.expand_dims(image, axis=0).astype(np.float32)
#     detector_interpreter.set_tensor(detector_input_details[0]['index'], image)
#     detector_interpreter.invoke()
#
#     # Extract detection results
#     output_data = detector_interpreter.get_tensor(detector_output_details[0]['index'])
#     results = []  # Populate this with detection results
#
#     # Process output data and convert it to readable format
#     for detection in output_data[0]:
#         confidence = detection[4]
#         if confidence > 0.5:  # Confidence threshold
#             class_id = int(detection[5])
#             box = detection[:4]
#             results.append({"class_id": class_id, "confidence": confidence, "box": box})
#
#     return results


import tensorflow as tf
import numpy as np
import cv2


def iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    Each box is given in the format [xmin, ymin, xmax, ymax].
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Compute the area of intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the Intersection over Union (IoU)
    iou_value = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou_value

def non_maximum_suppression(bboxes, class_indices, confidences, iou_threshold=0.5):
    """
    Applies Non-Maximum Suppression (NMS) to filter out overlapping bounding boxes.
    """
    indices = np.argsort(confidences)[::-1]  # Sort boxes by confidence score in descending order
    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        indices = indices[1:]

        # Suppress boxes with high IoU overlap
        filtered_indices = []
        for idx in indices:
            if iou(bboxes[current], bboxes[idx]) < iou_threshold:
                filtered_indices.append(idx)
        indices = np.array(filtered_indices)

    # Select boxes that are kept
    return bboxes[keep], [class_indices[i] for i in keep], [confidences[i] for i in keep]

def run_inference_tflite(image_path, model_path, input_size=(192, 192)):
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load and preprocess the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(image_rgb, input_size)
    input_image = input_image / 255.0  # Normalize to [0, 1]
    input_image = np.expand_dims(input_image, axis=0).astype(np.float32)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_image)

    # Run inference
    interpreter.invoke()

    # Process output data
    # Adjust according to your model's output structure (YOLOv5 typically has 3 output arrays)
    bboxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding boxes
    # class_indices = interpreter.get_tensor(output_details[1]['index'])[0]  # Class indices
    # scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores

    # Scale bounding boxes to original image dimensions
    h, w, _ = image.shape
    bboxes[:, [0, 2]] *= w  # Scale x-coordinates
    bboxes[:, [1, 3]] *= h  # Scale y-coordinates

    # Filter out detections with low confidence
    threshold = 0.5  # Set a confidence threshold
    bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2  # xmin
    bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2  # ymin
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]  # xmax
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]  # ymax
    bboxes[:, 5] = bboxes[:, 5].astype(int)
    filtered_bboxes, filtered_class_indices, filtered_confidences = non_maximum_suppression(
        bboxes[:, :4], bboxes[:, 5], bboxes[:, 4], iou_threshold=0.5
    )
    results = []
    for i, score in enumerate(filtered_bboxes):
        if filtered_confidences[i] >= threshold:
            bbox = filtered_bboxes[i]
            class_index = int(round(filtered_class_indices[i], 0))
            confidence = float(filtered_confidences[i])
            # results.append({
            #     'bbox': bbox,
            #     'class_index': class_index,
            #     'confidence': confidence
            # })
            results.append({"class_id": class_index, "confidence": confidence, "box": bbox})


    return results

# Usage example
# model_path = '../models/unisign_square_feature192_05122023.tflite'  # Path to your TFLite model
# image_path = '../data/1.jpg'  # Path to the image you want to test
#
# detections = run_inference_tflite(image_path, model_path)
# for detection in detections:
#     print(f"Bounding Box: {detection['bbox'][0:4]}, Class Index: {detection['class_index']}, Confidence: {detection['confidence']}")

